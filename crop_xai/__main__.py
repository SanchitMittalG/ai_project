import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reproducible XAI crop recommendation framework")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run training, tuning, evaluation, ablation and explanations")
    train.add_argument("--data", type=Path, default=Path("data/crop_recommendation.csv"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--jobs", type=int, default=1)
    train.add_argument("--cohort", choices=("paper22", "all"), default="paper22")
    train.add_argument("--quick", action="store_true", help="Reduced-grid smoke run, not final research results")
    train.add_argument("--no-explain", action="store_true")
    train.add_argument("--shap-samples", type=int, default=12)
    train.add_argument("--lime-samples", type=int, default=2500)
    predict = commands.add_parser("predict", help="Load only a trusted local joblib artifact")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", required=True, help="JSON object or path to a JSON file")
    predict.add_argument("--threshold", type=float, default=0.5)
    predict.add_argument("--top-k", type=int, default=3)
    predict.add_argument("--log", type=Path)
    predict.add_argument("--explain-output", type=Path, help="Export LIME and SHAP for this input")
    upload = commands.add_parser("upload", help="Create-only GCS backup using Application Default Credentials")
    upload.add_argument("--run", type=Path, required=True)
    upload.add_argument("--bucket", required=True)
    upload.add_argument("--prefix", required=True)
    args = parser.parse_args()
    try:
        if args.command == "train":
            from .experiment import run_experiment
            run_experiment(args.data, args.output, args.seed, args.folds, args.jobs, args.quick,
                           not args.no_explain, args.shap_samples, args.lime_samples, args.cohort)
        elif args.command == "predict":
            import joblib
            from .predict import append_prediction, recommend
            inputs = json.loads(args.input) if args.input.lstrip().startswith("{") else json.loads(
                Path(args.input).read_text(encoding="utf-8")
            )
            bundle = joblib.load(args.model)
            result = recommend(bundle, inputs, args.threshold, args.top_k)
            if args.explain_output:
                import pandas as pd
                from .explain import explain_model
                if args.explain_output.exists() and any(args.explain_output.iterdir()):
                    raise ValueError("Explanation output directory must be empty")
                explain_model(bundle, bundle["training_reference"], pd.DataFrame([inputs]),
                              args.explain_output, sample_count=1, sample_role="user input")
                result["explanation_directory"] = str(args.explain_output.resolve())
            if args.log:
                append_prediction(args.log, result)
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            from .cloud import upload_run
            print(json.dumps(upload_run(args.run, args.bucket, args.prefix), indent=2))
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
