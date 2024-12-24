from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging

from machine_predictive_maintenance.entity.config_entity import DataIngestionConfig
from machine_predictive_maintenance.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import pandas as pd
import numpy as np
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:

    def __init__(self, data_ingestion_config:DataIngestionConfig):
        """
        Initializes the DataIngestion class with the provided configuration.

        Parameters:
            data_ingestion_config: DataIngestionConfig
                Configuration object containing details for data ingestion.
        """
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
        

    def export_collection_as_dataframe(self):
        """
        Exports a MongoDB collection as a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing data from the specified MongoDB collection.

        Raises:
            MachinePredictiveMaintenanceException: If any error occurs during the data export.
        """
        try:
            database_name= self.data_ingestion_config.database_name
            collection_name= self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
        

    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        """
        Saves the provided DataFrame into a CSV file at the specified feature store location.

        Parameters:
            dataframe: pd.DataFrame
                The DataFrame to be saved as a CSV file.

        Returns:
            pd.DataFrame: The same DataFrame that was saved.

        Raises:
            MachinePredictiveMaintenanceException: If any error occurs during the file saving process.
        """
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e,sys)

    
    def split_data_as_train_test(self, dataframe:pd.DataFrame):
        """
        Splits the DataFrame into training and testing datasets and saves them as CSV files.

        Parameters:
            dataframe: pd.DataFrame
                The DataFrame to be split into training and testing datasets.

        Raises:
            MachinePredictiveMaintenanceException: If any error occurs during the data splitting or file saving process.
        """
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test file path.")

        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)

        
    def initiate_data_ingestion(self):
        """
        Executes the complete data ingestion process, including:
        1. Exporting data from MongoDB collection as a DataFrame.
        2. Saving the DataFrame to the feature store.
        3. Splitting the data into training and testing datasets and saving them.

        Returns:
            DataIngestionArtifact: An artifact containing paths for training and testing datasets.

        Raises:
            MachinePredictiveMaintenanceException: If any error occurs during the data ingestion process.
        """
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            dataingestionartifact= DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                         test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

            
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e,sys)
        