import pickle
from google.cloud.aiplatform.prediction.sklearn.predictor import SklearnPredictor


class AntiscamPredictor(SklearnPredictor):
    def __init__(self):
        """Initialize predictor"""
        import os
        os.environ["GOOGLE_CLOUD_PROJECT"] = "planar-pagoda-425919-f0"
        super().__init__()
        self._class_names = []
        self._preprocessor = None
        return

    def load(self, artifacts_uri: str):
        """Loads the preprocessor artifacts."""
        super().load(artifacts_uri)

        with open("preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        self._class_names = ["not_scam", "scam"]
        self._preprocessor = preprocessor

    def preprocess(self, prediction_input):
        """Perform scaling preprocessing"""
        inputs = super().preprocess(prediction_input)
        return self._preprocessor.transform(inputs)
