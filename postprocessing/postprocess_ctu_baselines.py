#!/usr/bin/env python3
"""Evaluate saved CTG baseline bundles on the private/public CTU cohorts."""
import argparse
import csv
from pathlib import Path

from baseline_model_bundle import evaluate_bundle, load_bundle
from run_ctg_fri_baseline import _risk_curve as fri_risk_curve
from run_xgboost_survival_baseline import _risk_curve as xgboost_risk_curve


BASELINES = {
    "ctg_fri_cox": (fri_risk_curve, "ctg_fri_cox"),
    "xgboost_ctg_cox": (xgboost_risk_curve, "xgboost_ctg"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", action="append", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--test-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="cache/baseline_landmarks_ctu")
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--lab-order-delay", type=float, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    statuses = []
    for model_file in args.model_file:
        try:
            # Read the small envelope first so the baseline-specific evaluator
            # is selected before applying the fitted model objects.
            import pickle
            with Path(model_file).open("rb") as handle:
                envelope = pickle.load(handle)
            baseline = envelope.get("baseline") if isinstance(envelope, dict) else None
            if baseline not in BASELINES:
                raise ValueError(f"Unsupported baseline bundle: {baseline!r}")
            risk_curve, stem = BASELINES[baseline]
            bundle = load_bundle(model_file, baseline=baseline)
            metrics_path = evaluate_bundle(
                bundle=bundle, trace_file=args.trace_file,
                test_label_file=args.test_labels, savedir=output / baseline,
                baseline=baseline, risk_curve=risk_curve, file_stem=stem,
                lab_order_delay=args.lab_order_delay,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_confidence_level=args.bootstrap_confidence_level,
                random_state=args.random_state, landmark_cache_dir=args.cache_dir,
            )
            statuses.append({"model_file": model_file, "baseline": baseline,
                             "status": "ok", "metrics_path": metrics_path, "error": ""})
        except Exception as error:  # report all bundles before returning failure
            statuses.append({"model_file": model_file, "baseline": "",
                             "status": "error", "metrics_path": "", "error": str(error)})
    status_path = output / "baseline_evaluation_status.csv"
    with status_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_file", "baseline", "status", "metrics_path", "error"])
        writer.writeheader()
        writer.writerows(statuses)
    if any(status["status"] != "ok" for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
