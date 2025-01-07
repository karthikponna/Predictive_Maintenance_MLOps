from machine_predictive_maintenance.components.data_ingestion import DataIngestion
from machine_predictive_maintenance.components.data_validation import DataValidation
from machine_predictive_maintenance.components.data_transformation import DataTransformation
from machine_predictive_maintenance.components.model_trainer import ModelTrainer 

from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging

from machine_predictive_maintenance.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from machine_predictive_maintenance.entity.config_entity import TrainingPipelineConfig

import sys

if __name__=="__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(data_ingestion_config)

        logging.info("Initiate the data ingestion")
        dataingestionartifact= data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)

        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, data_validation_config)

        logging.info("Initiate the data ingestion")
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)

        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Initiate the data transformation")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        

        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        print(model_trainer_artifact)

        logging.info("Model Training artifact created")



    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)
    