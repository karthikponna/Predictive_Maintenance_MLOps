import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
from sklearn.compose import ColumnTransformer

from machine_predictive_maintenance.constant.training_pipeline import TARGET_COLUMN

from machine_predictive_maintenance.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)
from machine_predictive_maintenance.constant.training_pipeline import SCHEMA_FILE_PATH
from machine_predictive_maintenance.entity.config_entity import DataTransformationConfig
from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging
from machine_predictive_maintenance.utils.main_utils.utils import read_yaml_file, drop_columns, save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise MachinePredictiveMaintenanceException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
        

    def get_data_transformer_object(self):

        try:

            logging.info("Got numerical cols from schema config")
            
            scaler = MinMaxScaler()
            
            # Fetching categories for OrdinalEncoder from schema config
            ordinal_categories = self._schema_config.get('ordinal_categories', [])
            ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)

            logging.info("Initialized MinMaxScaler, OrdinalEncoder with categories")

            ordinal_columns = self._schema_config['ordinal_columns']
            scaling_features = self._schema_config['scaling_features']

            preprocessor = ColumnTransformer(
                [
                    ("Ordinal_Encoder", ordinal_encoder, ordinal_columns),
                    ("MinMaxScaling", scaler, scaling_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            return preprocessor
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
        


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        

        try:
            
            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()


            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_train_df['Air temperature [c]'] = input_feature_train_df['Air temperature [K]'] - 273.15
            input_feature_train_df['Process temperature [c]'] = input_feature_train_df['Process temperature [K]'] - 273.15


            drop_cols = self._schema_config['drop_columns']

            input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)

            logging.info("Completed dropping the columns for Training dataset")
            


            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            input_feature_test_df['Air temperature [c]'] = input_feature_test_df['Air temperature [K]'] - 273.15
            input_feature_test_df['Process temperature [c]'] = input_feature_test_df['Process temperature [K]'] - 273.15


            drop_cols = self._schema_config['drop_columns']

            input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)

            logging.info("Completed dropping the columns for Testing dataset")

            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            
            
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            
            smt =  SMOTEENN(sampling_strategy="minority")

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

            logging.info("Applied SMOTEENN on training dataset")

            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )


            train_arr = np.c_[
                input_feature_train_final, np.array(target_feature_train_final)
            ]

            test_arr = np.c_[
                input_feature_test_final, np.array(target_feature_test_final)
            ]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor,)

            save_object( "final_model/preprocessor.pkl", preprocessor,)


            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact


        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)