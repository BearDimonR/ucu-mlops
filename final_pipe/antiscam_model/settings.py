from datetime import datetime

# runtime configs
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# general configs
REGION = "us-central1"
LOCATION = ""
PROJECT_ID = "planar-pagoda-425919-f0"
SERVICE_ACCOUNT = "475961067334-compute@developer.gserviceaccount.com"
BUCKET_NAME = "mlops-planar-pagoda-425919-f0-unique"
BUCKET_URI = "gs://mlops-planar-pagoda-425919-f0-unique"
# train configs
TRAIN_COMPUTE = "n1-standard-4"
TRAIN_IMAGE = "{}-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest".format(
    REGION.split("-")[0]
)
EXPERIMENT_NAME = f"ml-antiscam-auto-retrain"
MODEL_DIR = f"ml-antiscam/{TIMESTAMP}"
DATASET_ID = "ml"
TABLE_ID = "antiscam_train"
VALIDATE_TABLE_ID = "antiscam_val"
VALIDATE_STAGING_TABLE_ID = "antiscam_stg"
MODEL_NAME = "ml-antiscam"
# predictor configs
REPOSITORY = "ml-antiscam-prediction-container"
IMAGE = "antiscam-cpr-preprocess-server"
MODEL_ARTIFACT_DIR = f"{MODEL_DIR}/artifacts"
MODEL_DISPLAY_NAME = "ml-antiscam"
# local params
USER_SRC_DIR = "antiscam_model/predictor/src"
LOCAL_MODEL_ARTIFACTS_DIR = "model_artifacts"

SERVING_IMAGE_URI = "us-central1-docker.pkg.dev/planar-pagoda-425919-f0/ml-antiscam-prediction-container/antiscam-cpr-preprocess-server"
