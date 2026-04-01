"""
GreenIoT-MA — MLflow Tracking Helper
=======================================
Utilitaires pour le suivi des expériences MLflow.
"""

import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")


def init_mlflow(experiment_name: str = "greeniot_default"):
    """Initialise la connexion MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)
    print(f"   MLflow initialisé : {MLFLOW_URI}")
    print(f"   Expérience : {experiment_name}")
    return mlflow


def log_dataset_info(df, dataset_name: str = "dataset"):
    """Log les informations d'un dataset dans MLflow."""
    mlflow.log_params({
        f"{dataset_name}_rows": len(df),
        f"{dataset_name}_cols": len(df.columns),
        f"{dataset_name}_columns": str(list(df.columns)),
        f"{dataset_name}_dtypes": str(dict(df.dtypes)),
    })


def log_pipeline_metrics(metrics: dict, prefix: str = "pipeline"):
    """Log les métriques du pipeline de données."""
    for key, value in metrics.items():
        mlflow.log_metric(f"{prefix}_{key}", value)


def get_best_run(experiment_name: str, metric: str = "val_mae", ascending: bool = True):
    """Récupère le meilleur run d'une expérience."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    order = "ASC" if ascending else "DESC"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if runs.empty:
        return None

    return runs.iloc[0]


def list_experiments():
    """Liste toutes les expériences MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiments = mlflow.search_experiments()
    for exp in experiments:
        print(f"  [{exp.experiment_id}] {exp.name}")
    return experiments
