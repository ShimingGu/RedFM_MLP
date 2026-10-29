"""Label-blind, versioned interventions on Level-2 numerical metadata only."""
from __future__ import annotations

import re
import numpy as np

VERSION = 'filter_metadata_counterfactual_v1'
FIELDS = ('pivot_nm', 'width_nm')
RECORD = re.compile(r'(?m)^(filter=(?P<key>\S+) pivot_nm=)(?P<pivot>[0-9]+\.[0-9]{3})( width_nm=)(?P<width>[0-9]+\.[0-9]{3})(?= flux=)')
METADATA = re.compile(r' pivot_nm=\S+ width_nm=\S+')


def derangement(n, seed, forbidden=None):
    if n < 2:
        raise ValueError('A derangement needs at least two profiles')
    rng = np.random.default_rng(seed)
    for _ in range(100000):
        candidate = rng.permutation(n)
        if np.all(candidate != np.arange(n)) and (forbidden is None or np.all(candidate != forbidden)):
            return candidate
    raise ValueError('No reproducible derangement found')


def profile_groups(registry, releases):
    from .filter_photometry import filter_key
    hsc = [f'HSC.{b}' for b in ('g', 'r', 'i', 'z', 'y')]
    legacy = sorted({filter_key('desi-ls', b, int(r)) for r in releases for b in ('g', 'r', 'z', 'W1', 'W2')},
                    key=lambda k: (registry['filters'][k]['pivot_nm'], k))
    return {'hsc': hsc, 'legacy': legacy}


def build_conditions(registry, releases):
    groups = profile_groups(registry, releases)
    base = {k: {f: float(registry['filters'][k][f]) for f in FIELDS} for group in groups.values() for k in group}
    conditions = {}
    def add(name, family, rule, donor_indices=None, fields=FIELDS, factor=None):
        mapping = {k: dict(v) for k, v in base.items()}; donors = {}
        for gi, (label, keys) in enumerate(groups.items()):
            if donor_indices is not None:
                indices = donor_indices(len(keys), gi)
                donors[label] = {key: keys[int(j)] for key, j in zip(keys, indices)}
                for key, donor in donors[label].items():
                    for field in fields:
                        mapping[key][field] = base[donor][field]
            if factor is not None:
                for key in keys:
                    mapping[key]['pivot_nm'] *= factor
        conditions[name] = dict(family=family, rule=rule, mapping=mapping, donors=donors,
                                changed_fields=list(fields), units='nm', precision='%.3f')
    add('correct', 'control', 'Immutable original registry values', fields=())
    cycle = lambda n, gi: (np.arange(n)+1) % n
    reverse = lambda n, gi: (np.arange(n)-1) % n
    seeded = lambda n, gi: derangement(n, 1027031+gi*101)
    add('paired-cycle-forward', 'paired', 'Forward cycle within HSC and all actually used Legacy physical profiles; pair moved together', cycle)
    add('paired-cycle-reverse', 'paired', 'Reverse cycle in the same physical-profile groups; pair moved together', reverse)
    add('paired-derangement', 'paired', 'Derangement seed=1027031+101*domain_group_index; pair moved together', seeded)
    add('wavelength-cycle', 'wavelength', 'Same forward donor mapping as paired-cycle-forward; width stays correct', cycle, ('pivot_nm',))
    add('wavelength-derangement', 'wavelength', 'Same seeded donor mapping as paired-derangement; width stays correct', seeded, ('pivot_nm',))
    add('width-cycle', 'width', 'Same forward donor mapping as paired-cycle-forward; pivot stays correct', cycle, ('width_nm',))
    add('width-derangement', 'width', 'Same seeded donor mapping as paired-derangement; pivot stays correct', seeded, ('width_nm',))
    for name, factor in [('pivot-plus-02', 1.02), ('pivot-minus-02', .98), ('pivot-plus-05', 1.05), ('pivot-minus-05', .95)]:
        add(name, 'local-wavelength', f'Every physical-profile pivot multiplied by {factor}; width correct', fields=('pivot_nm',), factor=factor)
    # Optical versus infrared digit structure is kept, without using object measurements.
    mapping = {k: dict(v) for k, v in base.items()}; placebo_donors = {}
    strata = [groups['hsc'], [k for k in groups['legacy'] if not k.startswith('WISE.')],
              [k for k in groups['legacy'] if k.startswith('WISE.')]]
    for gi, keys in enumerate(strata):
        pivot = derangement(len(keys), 1027173+gi*101)
        width = derangement(len(keys), 1027291+gi*101, forbidden=pivot) if len(keys)>2 else pivot
        for i, key in enumerate(keys):
            pk, wk = keys[int(pivot[i])], keys[int(width[i])]
            mapping[key] = dict(pivot_nm=base[pk]['pivot_nm'], width_nm=base[wk]['width_nm'])
            placebo_donors[key] = dict(pivot_nm=pk, width_nm=wk)
    conditions['placebo-length-structure'] = dict(family='placebo', mapping=mapping, donors=placebo_donors,
        rule='Independent seeded pivot/width derangements within HSC, Legacy optical, and Legacy WISE strata. Distinct donors for pivot and width where n>2; the two WISE profiles swap both fields. Conserves registry marginals within strata, never physical identity associations. Seeds 1027173+101*stratum and 1027291+101*stratum. Token matching is measured, never assumed.',
        changed_fields=list(FIELDS), units='nm', precision='%.3f')
    return dict(version=VERSION, groups=groups, original=base, conditions=conditions,
                grouping='Each physical profile has one global mapping. Shared WISE profiles are never assigned inconsistent north/south values. DECam and BASS/MzLS remain distinct filter identities.')


def nonmetadata(text):
    return METADATA.sub('', text)


def intervene_text(text, mapping):
    """Use the original serializer's complete text; replace exactly two fields."""
    def replace(match):
        key = match['key']; values = mapping[key]
        if set(values) != set(FIELDS):
            raise ValueError('Only pivot_nm and width_nm can be intervened on')
        if not all(np.isfinite(values[f]) and values[f]>0 for f in FIELDS):
            raise ValueError('Finite positive metadata in nm required')
        return f"filter={key} pivot_nm={values['pivot_nm']:.3f} width_nm={values['width_nm']:.3f}"
    changed, n = RECORD.subn(replace, text)
    if n != sum(line.startswith('filter=') for line in text.splitlines()) or n==0:
        raise ValueError('Expected complete Level-2 flux records with three-decimal nm metadata')
    assert nonmetadata(changed) == nonmetadata(text)
    return changed
