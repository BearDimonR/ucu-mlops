import os

from google.cloud import aiplatform

from final_pipe.antiscam_model.settings import (
    PROJECT_ID,
    BUCKET_URI,
    MODEL_DIR,
    TIMESTAMP,
    TRAIN_IMAGE,
    EXPERIMENT_NAME,
    TRAIN_COMPUTE,
    SERVICE_ACCOUNT,
    DATASET_ID,
    TABLE_ID,
    MODEL_NAME,
    REGION,
    BUCKET_NAME,
    MODEL_DISPLAY_NAME,
    SERVING_IMAGE_URI,
)

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    experiment=EXPERIMENT_NAME,
    staging_bucket=BUCKET_URI,
)

job = aiplatform.CustomPythonPackageTrainingJob(
    display_name=f"{MODEL_NAME}-{TIMESTAMP}",
    python_package_gcs_uri=f"{BUCKET_URI}/trainer.tar.gz",
    python_module_name="src.task",
    container_uri=TRAIN_IMAGE,
)

CMDARGS = [
    "--model-dir=" + MODEL_DIR,
    "--bucket-name=" + BUCKET_NAME,
    f"--dataset-uri=bq://{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}",
    f"--experiment={EXPERIMENT_NAME}",
    f"--run=run-{TIMESTAMP}",
]

model = job.run(
    args=CMDARGS,
    replica_count=1,
    machine_type=TRAIN_COMPUTE,
    service_account=SERVICE_ACCOUNT,
    model_id=MODEL_NAME,
    is_default_version=True,
    sync=True,
)

aiplatform.Model.upload(
    serving_container_image_uri=SERVING_IMAGE_URI,
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
    serving_container_environment_variables={
        "GOOGLE_CLOUD_PROJECT": PROJECT_ID,
        "GCLOUD_PROJECT": PROJECT_ID,
        "project": PROJECT_ID,
        "location": REGION,
    },
    project=PROJECT_ID,
    location=REGION,
    model_id=MODEL_NAME,
    display_name=MODEL_DISPLAY_NAME,
    artifact_uri=os.path.join(BUCKET_URI, MODEL_DIR),
    sync=True,
    version_aliases=["production"],
)