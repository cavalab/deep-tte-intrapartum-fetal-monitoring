#!/usr/bin/env python3
"""Replay selected TTE checkpoints on CTU without fitting or fine-tuning."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

from train_time_to_event_models import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", action="append", required=True,
                        help="Checkpoint file, or directory containing selected .pt files.")
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--train-labels", required=True, help="HDF5 trace store containing labels")
    parser.add_argument("--test-labels", required=True, help="HDF5 trace store containing labels")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def checkpoint_paths(entries):
    paths = []
    for entry in entries:
        path = Path(entry)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("*.pt")))
        else:
            raise FileNotFoundError(path)
    return [path for path in paths if not path.name.endswith(("_optimizer.pt", "_training_config.pt"))]


def config_path(checkpoint):
    candidates = (
        checkpoint.with_name(checkpoint.stem + "_training_config.json"),
        checkpoint.parent.parent / (checkpoint.stem + "_training_config.json"),
    )
    return next((path for path in candidates if path.is_file()), None)


def trainer_kwargs(checkpoint, config, args):
    """Translate a training sidecar into an evaluation-only Trainer call."""
    mode = config.get("mode")
    if mode not in {"survival", "marked", "marked_corn", "competing", "competing_threshold"}:
        raise ValueError(
            "Checkpoint sidecar is not a supported time-to-event training config; "
            f"got mode={mode!r}."
        )
    model = dict(config.get("model") or {})
    if not model.get("model_class"):
        raise ValueError("Time-to-event training config is missing model.model_class.")
    fit_kwargs = config.get("fit_kwargs") or model.get("fit_kwargs") or {}
    if not isinstance(fit_kwargs, dict):
        raise TypeError("training config fit_kwargs must be a JSON object")
    # Optional encoder fields are often serialized as null; omitting them
    # reproduces each encoder's declared default more faithfully than passing
    # None through to an int()/float() conversion inside the constructor.
    fit_kwargs = {key: value for key, value in fit_kwargs.items() if value is not None}
    cuts = config.get("cuts")
    if not isinstance(cuts, list) or not cuts or any(value is None for value in cuts):
        raise ValueError("Time-to-event training config must contain finite checkpoint cuts.")
    durations = config.get("durations") or cuts
    if not isinstance(durations, list) or any(value is None for value in durations):
        durations = cuts
    digest = hashlib.sha256(str(checkpoint.resolve()).encode()).hexdigest()[:12]
    return {
        "trace_file": str(args.trace_file),
        "label_trainfile": str(args.train_labels),
        "label_testfile": str(args.test_labels),
        "splits_file": None,
        "savedir": str(args.output_dir),
        "mode": mode,
        "event_label": config.get("event_label") or "pH Cord < 7.1",
        "features": config["features"],
        "ml": model["model_class"],
        "chunk_window_size": int(config["chunk_window_size"]),
        "batch_size": int(config.get("batch_size") or 1024),
        "lab_order_delay": 0,
        "random_state": int(config.get("random_state") or 42),
        "durations": durations,
        "checkpoint_cuts": cuts,
        # Marked/survival sidecars correctly record competing-head fields as
        # null because their architecture does not use them. Trainer still
        # validates constructor arguments, so provide harmless defaults.
        "competing_head": model.get("competing_head") or "joint",
        "competing_head_hidden_dims": model.get("competing_head_hidden_dims") or [128, 64],
        "competing_head_batch_norm": (
            True if model.get("competing_head_batch_norm") is None
            else model["competing_head_batch_norm"]
        ),
        "competing_head_dropout": model.get("competing_head_dropout") or 0.2,
        "elapsed_time_feature": bool(config.get("elapsed_time_feature", False)),
        "elapsed_time_channel": bool(config.get("elapsed_time_channel", False)),
        "elapsed_time_scale_seconds": float(config.get("elapsed_time_scale_seconds") or 43200),
        "missingness_indicator_channels": bool(config.get("missingness_indicator_channels", False)),
        "missing_data_method": config.get("missing_data_method", "ffill"),
        "history_seconds": float(config.get("history_seconds") or 8 * 60 * 60),
        "chunk_missingness_max_fraction": config.get("chunk_missingness_max_fraction"),
        "fit_kwargs": fit_kwargs,
        # These settings affect fitting only, but Trainer validates them before
        # entering evaluation-only mode. Supply stable defaults for nullable
        # sidecar fields.
        "weight_decay": 0.0,
        "horizon": 0,
        "deephit_alpha": 0.2,
        "deephit_sigma": 0.1,
        "mark_loss_weight": 1.0,
        "missing_delivery_loss_weight": 1.0,
        "optimizer": "adafactor",
        "lr_schedule": "constant",
        "learning_rate": 0.01,
        "lr_warmup_fraction": 0.05,
        "lr_min_fraction": 0.01,
        "cocob_alpha": 100.0,
        "cocob_gradient_clip_norm": 1.0,
        "cocob_max_update_norm": 100.0,
        "cocob_max_parameter_abs": 1e4,
        "cocob_max_rejected_steps": 8,
        "early_stopping_patience": 12,
        "history_write_interval": 10,
        "training_landmark_diagnostic_patients": 0,
        "gpu": int(args.gpu),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_confidence_level": float(args.bootstrap_confidence_level),
        "eval_only": True,
        "checkpoint_path": str(checkpoint),
        "evaluation_name": f"ctu_external_{digest}",
        "skip_missing_checkpoint": False,
    }


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    for checkpoint in checkpoint_paths(args.checkpoint_dir):
        sidecar = config_path(checkpoint)
        if sidecar is None:
            errors.append(f"{checkpoint}: missing *_training_config.json")
            continue
        try:
            config = json.loads(sidecar.read_text())
            Trainer(**trainer_kwargs(checkpoint, config, args)).run()
        except Exception as error:
            errors.append(f"{checkpoint}: {error}\n{traceback.format_exc()}")
    if errors:
        message = "\n".join(errors)
        (output / "checkpoint_evaluation_errors.txt").write_text(message + "\n")
        if args.fail_on_error:
            raise SystemExit(message)


if __name__ == "__main__":
    main()
