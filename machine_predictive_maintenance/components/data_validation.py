from machine_predictive_maintenance.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from machine_predictive_maintenance.entity.config_entity import DataValidationConfig
from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging
from machine_predictive_maintenance.constant.training_pipeline import SCHEMA_FILE_PATH
from machine_predictive_maintenance.utils.main_utils.utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import sys, os


class DataValidation:

    """
    A class used to validate the dataset for machine predictive maintenance.

    This class includes methods to check the integrity of the dataset, 
    validate the number of columns, check if specific columns exist, and 
    detect data drift between the training and testing datasets.

    Attributes:
        data_ingestion_artifact (DataIngestionArtifact): Contains file paths for training and testing datasets.
        data_validation_config (DataValidationConfig): Configuration for data validation, including file paths for reports.
        _schema_config (dict): Schema configuration read from a YAML file, containing column specifications.
    """

    def __init__(self, data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        
        """
        Initializes the DataValidation object.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact containing file paths for training and testing datasets.
            data_validation_config (DataValidationConfig): Configuration for data validation.
        """

        try:
            self.data_ingestion_artifact= data_ingestion_artifact
            self.data_validation_config= data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:

        """
        Reads data from a CSV file into a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """

        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys) 

    
    def validate_number_of_columns(self, dataframe:pd.DataFrame)-> bool:

        """
        Validates if the number of columns in the DataFrame matches the schema.

        Args:
            dataframe (pd.DataFrame): DataFrame to validate.

        Returns:
            bool: True if the number of columns matches the schema, False otherwise.
        """

        try:
            number_of_columns = len(self._schema_config)

            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")

            if len(dataframe.columns) == number_of_columns: 
                return True
            
            return False
        
        except Exception as e:
            raise MachinePredictiveMaintenanceException (e, sys)
        

    def is_columns_exist(self, df:pd.DataFrame) -> bool:

        """
        Checks if all required columns exist in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to validate.

        Returns:
            bool: True if all required columns exist, False otherwise.
        """

        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

    
    def detect_dataset_drift(self, base_df, current_df, threshold = 0.05) -> bool :
        
        """
        Detects dataset drift between the base and current DataFrames using the Kolmogorov-Smirnov test.

        Args:
            base_df (pd.DataFrame): The base DataFrame.
            current_df (pd.DataFrame): The current DataFrame to compare.
            threshold (float, optional): Threshold for p-value to detect drift. Defaults to 0.05.

        Returns:
            bool: True if no drift is detected, False otherwise.
        """
        
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False

                else: 
                    is_found = True
                    status = False

                report.update({column:{
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
            
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)

        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)


        
    def initiate_data_validation(self)-> DataValidationArtifact:

        """
        Initiates the data validation process by validating column numbers, 
        column existence, and detecting dataset drift.

        Returns:
            DataValidationArtifact: An artifact containing validation results and file paths.
        """

        try:
            validation_error_msg  = ""
            logging.info("Starting data validation")

            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # read the data from the train and test 
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            logging.info(f"All required columns present in training dataframe: {status}")

            if not status:
                validation_error_msg += f"Train dataframe does not contain all columns.\n"

            status = self.validate_number_of_columns(dataframe=test_dataframe)

            if not status:
                validation_error_msg += f"Test dataframe does not contain all columns.\n"

            status = self.is_columns_exist(df=train_dataframe)

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."

            status = self.is_columns_exist(df=test_dataframe)

            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

                                  
            ## lets check datadrift
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True

            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            return data_validation_artifact
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e,sys)

