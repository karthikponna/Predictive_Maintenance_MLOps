import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo

from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

ca = certifi.where()

class PredictiveDataExtract():

    """
    A class to handle data extraction and insertion into MongoDB for predictive maintenance.

    Methods:
        csv_to_json_convertor(file_path): Converts a CSV file into a list of JSON records.
        insert_data_mongodb(records, database, collection): Inserts JSON records into the specified MongoDB database and collection.
    """

    def __init__(self):

        """
        Initializes the PredictiveDataExtract class.
        """

        try:
            pass
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
        
    def csv_to_json_convertor(self, file_path):

        """
        Converts a CSV file into a list of JSON records.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            list: A list of JSON records.

        """

        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
        
    def insert_data_mongodb(self, records, database, collection):

        """
        Inserts JSON records into the specified MongoDB database and collection.

        Args:
            records (list): List of JSON records to be inserted.
            database (str): Name of the MongoDB database.
            collection (str): Name of the MongoDB collection.

        Returns:
            int: Number of records inserted.
        """
        
        try:
            self.records = records
            self.database = database
            self.collection = collection

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise MachinePredictiveMaintenanceException(e, sys)
        
if __name__=="__main__":
    FILE_PATH="Machine_Predictive_Data/predictive_maintenance.csv"
    DATABASE="Predictive_Maintenance_MLOps"
    collection = "Machine_Predictive_Data"

    predictive_data_obj=PredictiveDataExtract()
    records = predictive_data_obj.csv_to_json_convertor(FILE_PATH)
    no_of_records = predictive_data_obj.insert_data_mongodb(records, DATABASE, collection)
    print(no_of_records)