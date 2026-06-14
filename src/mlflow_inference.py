import argparse
import json
import math

import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_MODEL_NAME = "my_model"
DEFAULT_ALIAS = "champion"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_TEST_FEATURES = "[[10.0, 4.0], [2.0, 3.0], [5.0, 1.0]]"
DEFAULT_TEST_TARGETS = "[46.0, 15.0, 29.0]"


def run_inference(
    model_name: str,
    alias: str,
    features: list[list[float]],
    targets: list[float] | None,
    tracking_uri: str,
    registry_uri: str | None,
    log_inference: bool,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri or tracking_uri)

    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(features)
    prediction_values = predictions.tolist()
    metrics = evaluate_regression_metrics(targets, predictions) if targets else {}

    if log_inference:
        mlflow.set_experiment("mlflow_inference_logs")
        with mlflow.start_run(run_name=f"inference_{model_name}_{alias}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_alias", alias)
            mlflow.log_param("model_uri", model_uri)
            mlflow.log_param("input_rows", len(features))
            mlflow.log_param("has_targets", bool(targets))
            mlflow.log_dict(
                {
                    "model_uri": model_uri,
                    "features": features,
                    "predictions": prediction_values,
                    "targets": targets,
                    "metrics": metrics,
                },
                "inference_payload.json",
            )
            if metrics:
                mlflow.log_metrics(metrics)
            run_id = mlflow.active_run().info.run_id
    else:
        run_id = None

    print(f"Loaded model: {model_uri}")
    print(f"Input features: {features}")
    print(f"Predictions: {prediction_values}")
    if targets:
        print(f"Targets: {targets}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
    if log_inference:
        print("Inference logged to MLflow experiment: mlflow_inference_logs")
        print(f"Inference run ID: {run_id}")


def evaluate_regression_metrics(targets: list[float], predictions: object) -> dict[str, float]:
    mse = mean_squared_error(targets, predictions)

    return {
        "inference_mae": mean_absolute_error(targets, predictions),
        "inference_rmse": math.sqrt(mse),
        "inference_r2_score": r2_score(targets, predictions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference using a model from the MLflow Model Registry."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Registered MLflow model name.",
    )
    parser.add_argument(
        "--alias",
        default=DEFAULT_ALIAS,
        help="Registered model alias to load, for example champion or champion_2.",
    )
    parser.add_argument(
        "--features",
        default="[[10.0, 4.0]]",
        help='Input features as JSON, for example "[[10.0, 4.0], [2.0, 3.0]]".',
    )
    parser.add_argument(
        "--targets",
        default=None,
        help='Optional expected regression targets as JSON, for example "[46.0, 15.0]".',
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run the built-in regression test data and print metrics.",
    )
    parser.add_argument(
        "--log-metrics",
        action="store_true",
        help="Deprecated. Inference logging is enabled by default.",
    )
    parser.add_argument(
        "--no-log-inference",
        action="store_true",
        help="Do not log this inference call to MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking store URI.",
    )
    parser.add_argument(
        "--registry-uri",
        default=None,
        help="MLflow registry store URI. Defaults to the tracking URI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = json.loads(DEFAULT_TEST_FEATURES if args.run_test else args.features)
    targets = json.loads(DEFAULT_TEST_TARGETS if args.run_test else args.targets) if (
        args.run_test or args.targets
    ) else None

    run_inference(
        model_name=args.model_name,
        alias=args.alias,
        features=features,
        targets=targets,
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
        log_inference=not args.no_log_inference,
    )


if __name__ == "__main__":
    main()
