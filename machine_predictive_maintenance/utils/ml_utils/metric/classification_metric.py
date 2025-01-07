from machine_predictive_maintenance.entity.artifact_entity import ClassificationMetricArtifact
from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:

    """
    Calculate classification metrics including F1 score, precision, and recall.

    Args:
        y_true (array-like): Ground truth (true labels).
        y_pred (array-like): Predicted labels.

    Returns:
        ClassificationMetricArtifact: An object containing F1 score, precision, and recall.

    """

    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score=precision_score(y_true,y_pred)

        classification_metric =  ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score, 
            recall_score=model_recall_score
        )
        
        return classification_metric
    except Exception as e:
        raise MachinePredictiveMaintenanceException(e,sys)