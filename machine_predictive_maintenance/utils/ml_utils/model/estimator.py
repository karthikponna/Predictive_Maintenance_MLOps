from machine_predictive_maintenance.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

import os
import sys

from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging


class MachinePredictiveModel:

    """
    A wrapper class for predictive models used in machine predictive maintenance.

    Attributes:
        model: The predictive model to be used for making predictions.

    Methods:
        predict(x): Predict the target variable based on the input features.

    Args:
        model: The predictive model instance.
    """

    def __init__(self, model):

        """
        Initialize the MachinePredictiveModel class.

        Args:
            model: The predictive model to be used.
        """

        try:
            self.model = model
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e,sys)
    
    def predict(self, x):

        """
        Predict the target variable using the model.

        Args:
            x (array-like): Input features for making predictions.

        Returns:
            array-like: Predicted values.

        """

        try:
            y_pred = self.model.predict(x)
            return y_pred
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e,sys)