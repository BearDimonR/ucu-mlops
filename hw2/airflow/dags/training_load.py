import pendulum
from airflow.decorators import dag
from airflow.models import Variable, Param
from airflow.operators.dummy import DummyOperator  # type: ignore
from airflow.operators.python import PythonOperator
import pandas as pd
from airflow.providers.google.cloud.hooks.gcs import GCSHook

default_task_args = {
    "owner": "DE_team",
    "depends_on_past": False,
    "retries": 0,
}


def load_training_data(**context) -> None:
    """
    Loads training data to the GCS
    """
    gcs_hook = GCSHook(gcp_conn_id="google_cloud_default")
    bucket_url = Variable.get(key="bucket_url")
    file_url = context["params"]["file_url"]

    if file_url.startswith("gs://"):
        source_bucket_name, source_blob_name = file_url[5:].split('/', 1)
        temp_file_path = "/tmp/training_data.csv"
        gcs_hook.download(bucket_name=source_bucket_name, object_name=source_blob_name, filename=temp_file_path)
    else:
        temp_file_path = file_url

    data = pd.read_csv(temp_file_path)

    temp_parquet_path = "/tmp/training_data.parquet"
    data.to_parquet(temp_parquet_path)
    gcs_hook.upload(bucket_name=bucket_url, object_name="test_dag/to/training_data.parquet", filename=temp_parquet_path)



@dag(
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, 1, 5),
    catchup=False,
    max_active_runs=1,
    default_args=default_task_args,
    tags=["ml"],
    params={
            "file_url": Param(
                "",
                type="string",
            )
        },
)
def load_training_data_main():
    """
    Load training data to the GCS
    @return: configured tasks for dag
    """
    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end")
    load_training_data_task = PythonOperator(
            task_id="load_training_data_task",
            python_callable=load_training_data,
            provide_context=True
        )

    start >> load_training_data_task >> end


load_training_data_main()
