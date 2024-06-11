import argparse
import os
import uuid

import google.cloud.aiplatform as aiplatform
import google.cloud.bigquery as bigquery
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()

parser.add_argument(
    "--experiment",
    dest="experiment",
    required=True,
    type=str,
    help="Name of experiment",
)
parser.add_argument(
    "--run",
    dest="run",
    required=True,
    type=str,
    help="Name of run within the experiment",
)

parser.add_argument(
    "--dataset-uri",
    dest="dataset_uri",
    required=True,
    type=str,
    help="Location of the dataset",
)
args = parser.parse_args()


def get_data(bq_client, dataset_uri, execution):
    dataset_artifact = aiplatform.Artifact.create(
        schema_title="system.Dataset", display_name="moderation_data", uri=dataset_uri
    )

    execution.assign_input_artifacts([dataset_artifact])

    prefix = "bq://"
    bq_table_uri = dataset_uri
    if bq_table_uri.startswith(prefix):
        bq_table_uri = bq_table_uri[len(prefix) :]
    table = bq_client.get_table(bq_table_uri)
    data = bq_client.list_rows(table).to_dataframe()

    X = data.drop("is_scam", axis=1)
    y = data["is_scam"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test


def get_model():
    params = {"class_weight": "balanced", "max_iter": 1000}
    model = LogisticRegression(**params)
    return model, params


def train_model(dataset, model, params):
    aiplatform.log_params(params)
    x_train_scaled, _, y_train, _ = dataset
    model.fit(x_train_scaled, y_train)
    return model


def evaluate_model(model, dataset):
    _, x_test_scaled, _, y_test = dataset
    y_pred = model.predict(x_test_scaled)

    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    metrics = {"f1": f1, "recall": recall, "roc_auc": roc_auc}
    return metrics


def save_model(model, model_dir, execution):
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/model.joblib"
    with open(model_path, "wb") as f:
        joblib.dump(model, f)

    model_artifact = aiplatform.Artifact.create(
        schema_title="system.Model", display_name="moderation_model", uri=model_dir
    )
    execution.assign_output_artifacts([model_artifact])


aiplatform.init(experiment=args.experiment)
aiplatform.start_run(args.run)
bq_client = bigquery.Client()


with aiplatform.start_execution(
    schema_title="system.ContainerExecution",
    display_name=f"{args.experiment}-{uuid.uuid1()}",
) as execution:
    dataset = get_data(bq_client, args.dataset_uri, execution)
    model, params = get_model()
    model = train_model(dataset, model, params)
    metrics = evaluate_model(model, dataset)
    save_model(model, args.model_dir, execution)

    aiplatform.log_metrics(
        {**metrics, "lineage": execution.get_output_artifacts()[0].lineage_console_uri}
    )

aiplatform.end_run()
