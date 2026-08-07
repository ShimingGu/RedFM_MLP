#!/usr/bin/env python3
"""Apply the archived Qwen QLoRA photo-z model to a magnitude CSV catalogue."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
except ImportError as exc:
    raise SystemExit(
        "Missing inference dependencies. Run this with the RedFM_MLP Pixi "
        "environment, or install torch, transformers, peft, bitsandbytes, "
        "pandas, and numpy."
    ) from exc


ARCHIVE_DIR = Path(
    "/arc/projects/ots/Cosmic_Imprint_of_Time/posttrainings/qwen-qlora"
)
SCRIPT_PARENT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = (
    SCRIPT_PARENT
    if (SCRIPT_PARENT / "adapter").is_dir()
    and (SCRIPT_PARENT / "photoz_head.pt").is_file()
    else ARCHIVE_DIR
)
DEFAULT_BASE_MODEL = Path(
    os.environ.get(
        "AION_QWEN_BASE",
        "/arc/home/gsm/hf_models/Qwen3.5-4B-Base",
    )
)

FEATURES = (
    "u_mag",
    "u_star_mag",
    "g_mag",
    "r_mag",
    "i_mag",
    "z_mag",
    "y_mag",
    "Y_mag",
    "J_mag",
    "H_mag",
    "Ks_mag",
)
FILLS = np.asarray(
    [
        25.977767944335938,
        25.803855895996094,
        25.46443271636963,
        25.128427505493164,
        24.86415672302246,
        24.64280128479004,
        24.341413497924805,
        24.810382843017578,
        24.58324432373047,
        24.365999221801758,
        24.095584869384766,
    ],
    dtype=np.float64,
)
TEXT_PREFIX = "galaxy all_magnitudes_ab schema=clauds_all_magnitude_v1"
N_Z_BINS = 300
Z_MIN = 0.005000030156224966
Z_MAX = 5.992499828338623
MAX_LENGTH = 2048
HEAD_HIDDEN_DIM = 256


class PhotoZHead(nn.Module):
    """The exact nonlinear PDF head used by the archived experiment."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HEAD_HIDDEN_DIM),
            nn.LayerNorm(HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HEAD_HIDDEN_DIM, HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HEAD_HIDDEN_DIM, N_Z_BINS),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


def hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    value = getattr(config, "hidden_size", None)
    if value is None:
        value = getattr(getattr(config, "text_config", None), "hidden_size", None)
    if value is None:
        base_config = getattr(getattr(model, "base_model", None), "config", None)
        value = getattr(getattr(base_config, "text_config", None), "hidden_size", None)
    if value is None:
        raise RuntimeError("Could not determine the Qwen hidden size.")
    return int(value)


class QwenPhotoZModel(nn.Module):
    def __init__(self, qwen: nn.Module):
        super().__init__()
        self.qwen = qwen
        self.photoz_head = PhotoZHead(hidden_size(qwen))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **model_inputs,
        )
        last_indices = attention_mask.sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(
            output.last_hidden_state.shape[0],
            device=output.last_hidden_state.device,
        )
        embedding = output.last_hidden_state[batch_indices, last_indices].float()
        return self.photoz_head(embedding)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV. Defaults beside the input catalogue.",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        help="Optional compressed NPZ containing pz and redshift_centers.",
    )
    parser.add_argument(
        "--id-column",
        help="Optional input identifier column copied into the PDF archive.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing adapter/ and photoz_head.pt.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=DEFAULT_BASE_MODEL,
        help="Matching local Qwen3.5-4B-Base checkpoint.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--missing-value",
        type=float,
        action="append",
        default=[],
        help="Finite missing sentinel to replace; repeat for multiple values.",
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable NF4 loading; requires substantially more memory.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to fetch files absent from the base checkpoint.",
    )
    parser.set_defaults(load_in_4bit=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive when provided.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    if not (args.artifact_dir / "adapter").is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {args.artifact_dir / 'adapter'}")
    if not (args.artifact_dir / "photoz_head.pt").is_file():
        raise FileNotFoundError(
            f"Photo-z head not found: {args.artifact_dir / 'photoz_head.pt'}"
        )
    if not args.base_model.exists() and not args.allow_download:
        raise FileNotFoundError(
            f"Base checkpoint not found: {args.base_model}. Set --base-model or "
            "AION_QWEN_BASE to the matching Qwen3.5-4B-Base checkpoint."
        )
    if args.id_column is not None and not args.id_column.strip():
        raise ValueError("--id-column cannot be empty.")


def load_input(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(args.input, nrows=args.max_rows)
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required magnitude columns: " + ", ".join(missing)
        )
    if args.id_column is not None and args.id_column not in frame.columns:
        raise ValueError(f"ID column not found: {args.id_column}")
    if frame.empty:
        raise ValueError("Input CSV contains no rows.")

    numeric = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    for sentinel in args.missing_value:
        values[values == float(sentinel)] = np.nan
    invalid = ~np.isfinite(values)
    if invalid.any():
        fill_matrix = np.broadcast_to(FILLS, values.shape)
        values[invalid] = fill_matrix[invalid]
    return frame, values


def serialize_rows(values: np.ndarray) -> list[str]:
    texts = []
    for row in values:
        fields = " ".join(
            f"{name}={float(value):.5f}"
            for name, value in zip(FEATURES, row, strict=True)
        )
        texts.append(f"{TEXT_PREFIX} {fields}")
    return texts


def load_model(args: argparse.Namespace, device: torch.device):
    if args.load_in_4bit and device.type != "cuda":
        raise RuntimeError("4-bit loading requires CUDA; use --no-4bit for CPU.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=not args.allow_download,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "local_files_only": not args.allow_download,
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        model_kwargs.update(
            {
                "dtype": torch.bfloat16,
                "device_map": {"": int(device_index)},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
            }
        )
    else:
        model_kwargs["dtype"] = (
            torch.bfloat16 if device.type == "cuda" else torch.float32
        )

    base = AutoModel.from_pretrained(args.base_model, **model_kwargs)
    if not args.load_in_4bit:
        base = base.to(device)
    qwen = PeftModel.from_pretrained(
        base,
        args.artifact_dir / "adapter",
        is_trainable=False,
    )
    model = QwenPhotoZModel(qwen).to(device)
    state = torch.load(
        args.artifact_dir / "photoz_head.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.photoz_head.load_state_dict(state)
    model.eval()
    return model, tokenizer


def redshift_centers() -> torch.Tensor:
    edges = torch.linspace(Z_MIN, Z_MAX, N_Z_BINS + 1, dtype=torch.float32)
    return 0.5 * (edges[:-1] + edges[1:])


def summarize_logits(
    logits: torch.Tensor,
    centers: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pz = torch.softmax(logits.float(), dim=-1)
    z_mean = torch.sum(pz * centers[None, :], dim=-1)
    z_mode = centers[torch.argmax(pz, dim=-1)]
    cdf = torch.cumsum(pz, dim=-1)

    def quantile(probability: float) -> torch.Tensor:
        indices = torch.argmax((cdf >= probability).to(torch.int64), dim=-1)
        return centers[indices]

    return {
        "pz": pz,
        "z_mean": z_mean,
        "z_mode": z_mode,
        "z_p16": quantile(0.16),
        "z_p50": quantile(0.50),
        "z_p84": quantile(0.84),
    }


def predict(
    texts: Sequence[str],
    *,
    model: nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int,
    keep_pdf: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    centers = redshift_centers()
    point_parts = {
        name: [] for name in ("z_mean", "z_mode", "z_p16", "z_p50", "z_p84")
    }
    pdf_parts = [] if keep_pdf else None

    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        encoded = tokenizer(
            list(texts[start:stop]),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            logits = model(**encoded)
        summary = summarize_logits(logits.detach().float().cpu(), centers)
        for name in point_parts:
            point_parts[name].append(summary[name].numpy())
        if pdf_parts is not None:
            pdf_parts.append(summary["pz"].numpy())
        if stop == len(texts) or start == 0 or stop % max(batch_size * 100, 1) == 0:
            print(f"predicted {stop:,}/{len(texts):,}", flush=True)

    points = {
        name: np.concatenate(parts).astype(np.float32, copy=False)
        for name, parts in point_parts.items()
    }
    pdf = (
        None
        if pdf_parts is None
        else np.concatenate(pdf_parts).astype(np.float32, copy=False)
    )
    return points, pdf


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    output = args.output or args.input.with_name(
        f"{args.input.stem}_qwen_qlora_photoz.csv"
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    frame, values = load_input(args)
    texts = serialize_rows(values)
    print(
        f"loading {args.artifact_dir} for {len(frame):,} rows on {device}",
        flush=True,
    )
    model, tokenizer = load_model(args, device)
    points, pdf = predict(
        texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        keep_pdf=args.pdf_output is not None,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame = frame.copy()
    for name, values_part in points.items():
        output_frame[name] = values_part
    output_frame.to_csv(output, index=False)
    print(f"saved {output}", flush=True)

    if args.pdf_output is not None:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pz": pdf,
            "redshift_centers": redshift_centers().numpy(),
        }
        if args.id_column is not None:
            payload["object_id"] = frame[args.id_column].astype(str).to_numpy()
        np.savez_compressed(args.pdf_output, **payload)
        print(f"saved {args.pdf_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
