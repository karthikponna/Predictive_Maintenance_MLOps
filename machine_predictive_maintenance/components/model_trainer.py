import os, sys

from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging

from machine_predictive_maintenance.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from machine_predictive_maintenance.entity.config_entity import ModelTrainerConfig

from machine_predictive_maintenance.utils.main_utils.utils import save_object, load_object
from machine_predictive_maintenance.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from machine_predictive_maintenance.utils.ml_utils.metric.classification_metric import get_classification_score
from machine_predictive_maintenance.utils.ml_utils.model.estimator import MachinePredictiveModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):

        """
        Initialize the ModelTrainer class with the provided artifacts and configuration.

        Args:
            data_transformation_artifact (DataTransformationArtifact): The artifact containing transformed training and testing data paths.
            model_trainer_config (ModelTrainerConfig): The configuration object for the model trainer, including file paths and other settings.
        """

        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
        
    def track_mlflow(self, best_model, classification_metric, input_example):

        """
        Log model and metrics to MLflow.

        Args:
            best_model: The trained model object.
            classification_metric: The classification metrics (f1, precision, recall).
            input_example: An example input data sample for the model.
        """

        try:
            with mlflow.start_run():
                f1_score=classification_metric.f1_score
                precision_score=classification_metric.precision_score
                recall_score=classification_metric.recall_score

                mlflow.log_metric("f1_score",f1_score)
                mlflow.log_metric("precision",precision_score)
                mlflow.log_metric("recall_score",recall_score)
                mlflow.sklearn.log_model(best_model,"model", input_example=input_example)
                
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
    

    def train_model(self, X_train, y_train, X_test, y_test ):

        """
        Train multiple models and select the best-performing one based on evaluation metrics.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Testing features.
            y_test: Testing labels.

        Returns:
            ModelTrainerArtifact: An artifact containing details about the trained model and its metrics.
        """

        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }

        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                             models=models, param=params)
        
        best_model_score = max(sorted(model_report.values()))

        logging.info(f"Best Model Score: {best_model_score}")

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        
        logging.info(f"Best Model Name: {best_model_name}")

        best_model = models[best_model_name]

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        input_example = X_train[:1]
        print(input_example)
        # Track the experiments with mlflow
        self.track_mlflow(best_model, classification_train_metric, input_example)


        y_test_pred=best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # Track the experiments with mlflow
        self.track_mlflow(best_model, classification_test_metric, input_example)


        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)


        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Machine_Predictive_Model = MachinePredictiveModel(model=best_model)

        save_object(self.model_trainer_config.trained_model_file_path,obj=MachinePredictiveModel)

        save_object("final_model/model.pkl",best_model)

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        """
        Prepare and train the model using the transformed data.

        Returns:
            ModelTrainerArtifact: An artifact containing the path of the trained model and evaluation metrics.
        """

        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(X_train,y_train,X_test,y_test)
            return model_trainer_artifact
            

        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)