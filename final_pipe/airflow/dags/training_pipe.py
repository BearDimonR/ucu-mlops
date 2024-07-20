import logging
import os
from datetime import datetime, timedelta
import re

import pandas as pd
import pendulum
from airflow.decorators import dag
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator  # type: ignore
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
import tempfile
from airflow.models import XCom

from airflow.utils.task_group import TaskGroup
from gcloud.aio.bigquery import bigquery
from google.cloud import aiplatform
from google.cloud.bigquery import LoadJobConfig, WriteDisposition, SourceFormat

from dags.settings import (
    PROJECT_ID,
    LOCATION,
    BUCKET_URI,
    MODEL_DIR,
    TIMESTAMP,
    TRAIN_IMAGE,
    EXPERIMENT_NAME,
    TRAIN_COMPUTE,
    SERVICE_ACCOUNT,
    DATASET_ID,
    TABLE_ID,
    VALIDATE_TABLE_ID,
    VALIDATE_STAGING_TABLE_ID,
    MODEL_NAME,
    SERVING_IMAGE_URI,
    MODEL_DISPLAY_NAME,
    BUCKET_NAME,
)


def extract_data_from_bq(**context):
    extract_date_str = Variable.get(key="data_source_date", default_var="2023-01-01")
    extract_date = datetime.strptime(extract_date_str, "%Y-%m-%d")
    new_extract_date = extract_date + timedelta(days=30)
    new_extract_date_str = new_extract_date.strftime("%Y-%m-%d")
    # define hook
    bq_hook = BigQueryHook(bigquery_conn_id="bigquery_default", use_legacy_sql=False)
    sql = """
    SELECT *
    FROM `ml.data_source` WHERE DATE(event_timestamp) >= DATE("%s") AND DATE(event_timestamp) < DATE("%s")
    """ % (
        extract_date_str,
        new_extract_date_str,
    )
    # get batch of data
    df = bq_hook.get_pandas_df(sql=sql)

    # store file to parquet
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    df.to_parquet(tmpfile.name, index=False)
    context["ti"].xcom_push(key="extracted_file_path", value=tmpfile.name)
    # adjust variable
    Variable.set(key="data_source_date", value=new_extract_date_str)


def transform_data(**context):
    # get file
    extracted_file_path = context["ti"].xcom_pull(
        key="extracted_file_path", task_ids="data_pipeline.extract_data_from_source"
    )
    df = pd.read_parquet(extracted_file_path)
    # apply transformations
    columns_to_drop = [
        # redundant columns
        "antiscam_cases__is_scam",
        # not for prediction
        "event_timestamp",
        "antiscam_cases__trigger_type",
    ]

    # filter only 2023 year
    df_clean = df[(df["event_timestamp"] > "2023-01-01")]
    # fill numeric columns with zero values (double check whether only required columns have zeros)
    df_clean = df_clean.drop(columns=columns_to_drop).fillna(0)

    # rename columns to remove prefix
    df_clean = df_clean.rename(
        columns={name: re.sub(r"[a-z_]+__", "", name) for name in df_clean.columns}
    )

    # make fraction
    df_val = df_clean.sample(frac=0.02)
    df_train = df_clean.drop(df_val.index)

    # upload adjusted file
    tmpfile_train = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmpfile_val = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")

    # save files
    df_train.to_parquet(tmpfile_train.name, index=False)
    df_val.to_parquet(tmpfile_val.name, index=False)
    context["ti"].xcom_push(
        key="transformed_file_paths", value=[tmpfile_train.name, tmpfile_val.name]
    )


def load_to_bigquery(**context):
    # get file path
    ti = context["ti"]
    transformed_file_paths = ti.xcom_pull(
        key="transformed_file_paths", task_ids="data_pipeline.transform_data"
    )

    # get bq client
    bq_hook = BigQueryHook(bigquery_conn_id="bigquery_default", use_legacy_sql=False)
    client = bq_hook.get_client()

    # create load job
    job_config = LoadJobConfig(
        source_format=SourceFormat.PARQUET,
        autodetect=True,
        write_disposition=WriteDisposition.WRITE_APPEND,
    )

    # run job
    with open(transformed_file_paths[0], "rb") as file:
        load_job = client.load_table_from_file(
            file, f"{DATASET_ID}.{TABLE_ID}", job_config=job_config
        )

    with open(transformed_file_paths[1], "rb") as file:
        load_job_validate = client.load_table_from_file(
            file, f"{DATASET_ID}.{VALIDATE_TABLE_ID}", job_config=job_config
        )

    # wait for job
    load_job.result()
    load_job_validate.result()
    logging.info(
        f"Loaded {transformed_file_paths} to BigQuery tables {DATASET_ID}.{TABLE_ID} and {DATASET_ID}.{VALIDATE_TABLE_ID}"
    )


def create_training_job(**context):
    timestamp = pendulum.instance(context["ti"].execution_date).strftime(
        "%Y-%m-%d-%H-%M-%S"
    )
    bq_hook = BigQueryHook(bigquery_conn_id="bigquery_default", use_legacy_sql=False)

    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        experiment=EXPERIMENT_NAME,
        staging_bucket=BUCKET_URI,
        service_account=SERVICE_ACCOUNT,
        credentials=bq_hook.get_credentials(),
    )

    model = aiplatform.Model(model_name=MODEL_NAME, version="production")

    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=f"{MODEL_NAME}-{timestamp}",
        python_package_gcs_uri=f"{BUCKET_URI}/trainer.tar.gz",
        python_module_name="src.task",
        container_uri=TRAIN_IMAGE,
    )

    job.run(
        args=[
            "--model-dir=" + MODEL_DIR,
            "--bucket-name=" + BUCKET_NAME,
            f"--dataset-uri=bq://{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}",
            f"--experiment={EXPERIMENT_NAME}",
            f"--run=run-{timestamp}",
        ],
        replica_count=1,
        machine_type=TRAIN_COMPUTE,
        service_account=SERVICE_ACCOUNT,
        sync=True,
    )

    model = aiplatform.Model.upload(
        serving_container_image_uri=SERVING_IMAGE_URI,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        model_id=MODEL_NAME,
        display_name=MODEL_DISPLAY_NAME,
        parent_model=model.name,
        artifact_uri=os.path.join(BUCKET_URI, MODEL_DIR),
        sync=True,
        is_default_version=False,
        version_aliases=["auto-training"],
    )

    return model.versioned_resource_name, f"run-{timestamp}"


def evaluate_model(**context):
    model_name, run_id = context["ti"].xcom_pull(
        task_ids="training_pipeline.create_training_job"
    )

    bq_hook = BigQueryHook(bigquery_conn_id="bigquery_default", use_legacy_sql=False)

    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        experiment=EXPERIMENT_NAME,
        staging_bucket=BUCKET_URI,
        service_account=SERVICE_ACCOUNT,
        credentials=bq_hook.get_credentials(),
    )

    df = aiplatform.get_experiment_df(EXPERIMENT_NAME)
    print(df)
    run_recall = df[df["run_name"] == run_id]["metric.recall"].values[0]
    has_largest_recall = run_recall == df["metric.recall"].max()

    # load model
    model = aiplatform.Model(model_name=model_name)
    evaluation_job = model.evaluate(
        class_labels=["scam", "not_scam"],
        prediction_type="classification",
        target_field_name="is_scam",
        bigquery_source_uri=f"bq://{PROJECT_ID}.{DATASET_ID}.{VALIDATE_TABLE_ID}",
        bigquery_destination_output_uri=f"bq://{PROJECT_ID}.{DATASET_ID}.{VALIDATE_STAGING_TABLE_ID}-current",
        service_account=SERVICE_ACCOUNT,
    )

    # previous
    previous_model = aiplatform.Model(model_name=MODEL_NAME, version="production")
    previous_evaluation_job = previous_model.evaluate(
        class_labels=["scam", "not_scam"],
        prediction_type="classification",
        target_field_name="is_scam",
        bigquery_source_uri=f"bq://{PROJECT_ID}.{DATASET_ID}.{VALIDATE_TABLE_ID}",
        bigquery_destination_output_uri=f"bq://{PROJECT_ID}.{DATASET_ID}.{VALIDATE_STAGING_TABLE_ID}-prev",
        service_account=SERVICE_ACCOUNT,
    )

    return (
        "training_pipeline.deploy_model"
        if has_largest_recall
        else "training_pipeline.skip_deployment"
    )


def deploy_model(**context):
    model_name, run_id = context["ti"].xcom_pull(
        task_ids="training_pipeline.create_training_job"
    )
    # deploy model
    bq_hook = BigQueryHook(bigquery_conn_id="bigquery_default", use_legacy_sql=False)

    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        experiment=EXPERIMENT_NAME,
        staging_bucket=BUCKET_URI,
        service_account=SERVICE_ACCOUNT,
        credentials=bq_hook.get_credentials(),
    )
    model = aiplatform.Model(model_name=model_name)
    endpoint = aiplatform.Endpoint.list(
        filter=f"display_name={MODEL_NAME}", order_by="create_time"
    )[0]

    if len(endpoint.gca_resource.deployed_models) > 0:
        deployed_model_id = endpoint.gca_resource.deployed_models[0].id
        endpoint.undeploy(deployed_model_id)

    endpoint.deploy(
        model=model,
        deployed_model_display_name="ml-antiscam",
        machine_type=TRAIN_COMPUTE,
        sync=False,
    )
    # update aliases
    model.versioning_registry.add_version_aliases(
        new_aliases=["default", "production"], version=model.version_id
    )

    return endpoint.raw_predict_request_url


@dag(
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, 1, 5),
    catchup=False,
    max_active_runs=1,
    tags=["ml"],
)
def load_training_data_main():
    """
    Load training data to the GCS
    @return: configured tasks for dag
    """
    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end")
    with TaskGroup(
        "data_pipeline", tooltip="Data Extraction, Transformation, and Loading Pipeline"
    ) as data_pipeline:
        extract_data_task = PythonOperator(
            task_id="extract_data_from_source",
            python_callable=extract_data_from_bq,
            provide_context=True,
        )

        transform_data_task = PythonOperator(
            task_id="transform_data",
            python_callable=transform_data,
            provide_context=True,
        )

        load_to_bigquery_task = PythonOperator(
            task_id="load_to_bigquery",
            python_callable=load_to_bigquery,
            provide_context=True,
        )

        extract_data_task >> transform_data_task >> load_to_bigquery_task

    with TaskGroup(
        "training_pipeline", tooltip="Training, Fine-tuning, Evaluating"
    ) as training_pipeline:
        create_training_task = PythonOperator(
            task_id="create_training_job",
            python_callable=create_training_job,
            provide_context=True,
        )

        evaluate_model_task = BranchPythonOperator(
            task_id="evaluate_model",
            python_callable=evaluate_model,
            provide_context=True,
        )

        deploy_model_task = PythonOperator(
            task_id="deploy_model",
            python_callable=deploy_model,
            provide_context=True,
        )

        skip_deployment_task = DummyOperator(task_id="skip_deployment")

        (
            create_training_task
            >> evaluate_model_task
            >> [deploy_model_task, skip_deployment_task]
        )

    (start >> data_pipeline >> training_pipeline >> end)


load_training_data_main()
