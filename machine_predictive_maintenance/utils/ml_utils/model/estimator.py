from machine_predictive_maintenance.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

import os
import sys

from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging


class MachinePredictiveModel:
    def __init__(self, model):
        try:
            self.model = model
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e,sys)
    
    def predict(self, x):
        try:
            y_pred = self.model.predict(x)
            return y_pred
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e,sys)