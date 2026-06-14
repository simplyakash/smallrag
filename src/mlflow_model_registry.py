import argparse
import json
import math
import pickle
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_MODEL_NAME = "registered_model"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def register_saved_sklearn_model(
    model_name: str,
    model_path: Path,
    tracking_uri: str,
    registry_uri: str | None,
    alias: str | None,
    artifact_path: str,
    eval_features: list[list[float]] | None,
    eval_targets: list[float] | None,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri or tracking_uri)
    mlflow.set_experiment("mlflow_register_existing_model")

    model = load_saved_model(model_path)

    with mlflow.start_run(run_name=f"register_{model_name}") as run:
        mlflow.log_param("source_model_path", str(model_path))
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.set_tag("use_case", "register_existing_model")

        metrics = {}
        if eval_features and eval_targets:
            metrics = evaluate_regression_model(model, eval_features, eval_targets)
            mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

        client = MlflowClient()
        model_version = _get_logged_model_version(
            client=client,
            model_name=model_name,
            run_id=run.info.run_id,
        )

        if alias:
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=model_version,
            )

        print(f"Run ID: {run.info.run_id}")
        print(f"Source model path: {model_path}")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Registered model: {model_name}")
        print(f"Registered version: {model_version}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        if alias:
            print(f"Alias set: models:/{model_name}@{alias}")


def register_model_uri(
    model_name: str,
    model_uri: str,
    tracking_uri: str,
    registry_uri: str | None,
    alias: str | None,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri or tracking_uri)

    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    if alias:
        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=registered_model.version,
        )

    print(f"Source model URI: {model_uri}")
    print(f"Registered model: {model_name}")
    print(f"Registered version: {registered_model.version}")
    if alias:
        print(f"Alias set: models:/{model_name}@{alias}")


def load_saved_model(model_path: Path) -> object:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.suffix == ".joblib":
        return joblib.load(model_path)

    if model_path.suffix in {".pkl", ".pickle"}:
        with model_path.open("rb") as model_file:
            return pickle.load(model_file)

    raise ValueError("Use a .pkl, .pickle, or .joblib sklearn model file.")


def evaluate_regression_model(
    model: object,
    eval_features: list[list[float]],
    eval_targets: list[float],
) -> dict[str, float]:
    predictions = model.predict(eval_features)
    mse = mean_squared_error(eval_targets, predictions)

    return {
        "mae": mean_absolute_error(eval_targets, predictions),
        "rmse": math.sqrt(mse),
        "r2_score": r2_score(eval_targets, predictions),
    }


def _get_logged_model_version(
    client: MlflowClient,
    model_name: str,
    run_id: str,
) -> str:
    versions = client.search_model_versions(f"name = '{model_name}'")
    for version in versions:
        if version.run_id == run_id:
            return version.version

    raise RuntimeError(f"Could not find a registered version for run {run_id}.")


def load_registered_model(
    model_name: str,
    tracking_uri: str,
    registry_uri: str | None,
    alias: str,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri or tracking_uri)

    model_uri = f"models:/{model_name}@{alias}"
    mlflow.pyfunc.load_model(model_uri)

    print(f"Loaded registered model: {model_uri}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register an already trained model with the MLflow Model Registry."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Registered model name to create or update.",
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
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to an already trained sklearn model file (.pkl, .pickle, or .joblib).",
    )
    parser.add_argument(
        "--model-uri",
        default=None,
        help="Existing MLflow model URI to register, for example runs:/<run_id>/model.",
    )
    parser.add_argument(
        "--artifact-path",
        default="model",
        help="Artifact path to use when logging a saved sklearn model file.",
    )
    parser.add_argument(
        "--eval-features",
        default=None,
        help='Optional regression eval features as JSON, for example "[[10.0, 4.0]]".',
    )
    parser.add_argument(
        "--eval-targets",
        default=None,
        help='Optional regression eval targets as JSON, for example "[46.0]".',
    )
    parser.add_argument(
        "--alias",
        default="champion",
        help="Registry alias to set on the new model version. Use an empty value to skip.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the registered model alias instead of registering a new version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tracking_uri.startswith("file:"):
        Path(args.tracking_uri.removeprefix("file:")).mkdir(parents=True, exist_ok=True)

    alias = args.alias or None
    eval_features = json.loads(args.eval_features) if args.eval_features else None
    eval_targets = json.loads(args.eval_targets) if args.eval_targets else None

    if bool(eval_features) != bool(eval_targets):
        raise ValueError("Provide both --eval-features and --eval-targets, or neither.")

    if args.load:
        load_registered_model(
            args.model_name,
            args.tracking_uri,
            args.registry_uri,
            args.alias or "champion",
        )
        return

    if bool(args.model_path) == bool(args.model_uri):
        raise ValueError("Provide exactly one of --model-path or --model-uri.")

    if args.model_path:
        register_saved_sklearn_model(
            model_name=args.model_name,
            model_path=args.model_path,
            tracking_uri=args.tracking_uri,
            registry_uri=args.registry_uri,
            alias=alias,
            artifact_path=args.artifact_path,
            eval_features=eval_features,
            eval_targets=eval_targets,
        )
        return

    register_model_uri(
        model_name=args.model_name,
        model_uri=args.model_uri,
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
        alias=alias,
    )


if __name__ == "__main__":
    main()
