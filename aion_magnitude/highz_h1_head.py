"""Isolated H1 three-component globally truncated Gaussian-mixture control.

Returns normalized log *bin masses*, consumed as logits by the unchanged RedFM
CE/softmax evaluator. CDF arithmetic is float64; backbone/head trunk stay fp32.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .models import PhotoZHead
from .utils import make_redshift_grid

COMPONENTS = 3
MIN_SCALE = 0.0075  # half a native 0.015-wide redshift bin, fixed before P scoring
MAX_SCALE = 4.5     # full modeled support width; permits very diffuse predictions
INITIAL_SCALE = 0.3


def normal_log_interval(a, b):
    """log[Phi(b)-Phi(a)], using the upper tail when both bounds are positive.

Reflection avoids subtracting CDFs rounded to one. expm1 avoids cancellation.
The float64 epsilon guard only prevents a zero numerical difference; on the
bounded H1 mean/scale domain finite intervals have much larger differences.
"""
    use_survival = (a + b) > 0
    upper = torch.where(use_survival, -a, b)
    lower = torch.where(use_survival, -b, a)
    log_upper = torch.special.log_ndtr(upper)
    log_lower = torch.special.log_ndtr(lower)
    ratio = (log_lower - log_upper).clamp_max(-torch.finfo(a.dtype).eps)
    return log_upper + torch.log(-torch.expm1(ratio))


def parameters_from_raw(raw):
    raw = raw.to(torch.float64)
    log_weights = torch.log_softmax(raw[..., :3], dim=-1)
    means = 4.5 * torch.sigmoid(raw[..., 3:6])
    scales = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * torch.sigmoid(raw[..., 6:9])
    return log_weights, means, scales


def mixture_log_bin_masses(raw, edges):
    weights, means, scales = parameters_from_raw(raw)
    edges = edges.to(device=raw.device, dtype=torch.float64)
    a = (edges[:-1] - means[..., :, None]) / scales[..., :, None]
    b = (edges[1:] - means[..., :, None]) / scales[..., :, None]
    component = normal_log_interval(a, b)
    mass = torch.logsumexp(weights[..., :, None] + component, dim=-2)
    # This is one global mixture truncation, not separately normalized components.
    return mass - torch.logsumexp(mass, dim=-1, keepdim=True)


class TruncatedGaussianMixtureHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        # Identical shared hidden layers and state names, output is 9 rather than 300.
        self.net = PhotoZHead(input_dim, 9, hidden_dim).net
        edges, _ = make_redshift_grid(0., 4.5, 300)
        self.register_buffer('redshift_edges', edges, persistent=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(421102)
            nn.init.normal_(self.net[6].weight, mean=0., std=0.01)
            with torch.no_grad():
                self.net[6].bias.zero_()
                self.net[6].bias[3:6].copy_(torch.logit(torch.tensor([1/6, .5, 5/6])))
                scale = (INITIAL_SCALE-MIN_SCALE)/(MAX_SCALE-MIN_SCALE)
                self.net[6].bias[6:9].fill_(math.log(scale/(1-scale)))

    def reset_source_diagnostics(self):
        self._source_diagnostics = None

    def source_diagnostics(self):
        value = getattr(self, '_source_diagnostics', None)
        if value is None: return {}
        n = value['n']
        return dict(source_objects=n, component_mean_min=value['mu_min'], component_mean_max=value['mu_max'],
            component_scale_min=value['sigma_min'], component_scale_max=value['sigma_max'],
            mean_component_scales=(value['sigma_sum']/n).tolist(), mean_component_means=(value['mu_sum']/n).tolist(),
            minimum_scale=MIN_SCALE, maximum_scale=MAX_SCALE)

    def raw_parameters(self, x):
        return self.net(x)

    def forward(self, x):
        raw = self.raw_parameters(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            if not self.training:
                with torch.no_grad():
                    _,mu,sigma=parameters_from_raw(raw.detach())
                    current=dict(n=len(mu),mu_min=float(mu.min()),mu_max=float(mu.max()),sigma_min=float(sigma.min()),sigma_max=float(sigma.max()),
                        mu_sum=mu.sum(0).cpu(),sigma_sum=sigma.sum(0).cpu())
                    previous=getattr(self,'_source_diagnostics',None)
                    if previous is not None:
                        current['n']+=previous['n'];current['mu_sum']+=previous['mu_sum'];current['sigma_sum']+=previous['sigma_sum']
                        for k in ['mu_min','sigma_min']:current[k]=min(current[k],previous[k])
                        for k in ['mu_max','sigma_max']:current[k]=max(current[k],previous[k])
                    self._source_diagnostics=current
            return mixture_log_bin_masses(raw, self.redshift_edges).float()


def replace_head(model):
    """Copy every shape-compatible canonical trunk tensor, never output tensors."""
    old = model.photoz_head
    device = next(old.parameters()).device
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(421102)
        head = TruncatedGaussianMixtureHead(old.net[0].in_features, old.net[0].out_features)
    state = old.state_dict()
    own = head.state_dict()
    shared = {k:v for k,v in state.items() if k in own and v.shape == own[k].shape}
    assert set(shared) == {k for k in own if not k.startswith('net.6.')}
    head.load_state_dict({**own, **shared}, strict=True)
    model.photoz_head = head.to(device)
    for k in shared:
        torch.testing.assert_close(model.photoz_head.state_dict()[k], state[k], rtol=0, atol=0)
    return model
