import os
from google.cloud import aiplatform
from google.cloud.aiplatform import Endpoint
from google.cloud.aiplatform.prediction import LocalModel

from final_pipe.antiscam_model.settings import (
    REGION,
    PROJECT_ID,
    REPOSITORY,
    IMAGE,
    USER_SRC_DIR,
    MODEL_NAME
)
from src.predictor import AntiscamPredictor

aiplatform.init(project=PROJECT_ID, location=REGION)

local_model = LocalModel.build_cpr_model(
    USER_SRC_DIR,
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}",
    predictor=AntiscamPredictor,
    requirements_path=os.path.join(USER_SRC_DIR, "requirements.txt"),
)

serving_container = local_model.get_serving_container_spec()
local_model.push_image()

# endpoint = Endpoint.create(
#     display_name=MODEL_NAME,
#     project=PROJECT_ID,
#     location=REGION,
# )

#print(serving_container)
#print(endpoint)
