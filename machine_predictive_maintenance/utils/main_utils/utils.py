import yaml
from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging
import os, sys
import numpy as np
import pandas as pd
# import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            yaml.dump(content, file)
    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            np.save(file_path, array)
    
    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)
    
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
        
    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)


def load_object(file_path: str, ) -> object:

    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)
    


def drop_columns(df: pd.DataFrame, cols: list)-> pd.DataFrame:

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped 
    """
    logging.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):

    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=5)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)
    
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from machine_predictive_maintenance.utils.main_utils.utils import read_yaml_file, drop_columns
from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging
import numpy as np


def processing_test_data(data: pd.DataFrame, schema_file: str, preprocessor: BaseEstimator) -> pd.DataFrame:
    """
    Transforms the raw test data using the schema file and preprocessor.

    Args:
        data (pd.DataFrame): The raw input data.
        schema_file (str): Path to the schema file (YAML).
        preprocessor (BaseEstimator): Pretrained preprocessing object (e.g., OrdinalEncoder, MinMaxScaler).

    Returns:
        pd.DataFrame: Transformed test data ready for prediction.
    """
    try:
        # Load schema configuration
        schema = read_yaml_file(schema_file)
        df = data.copy()

        # Temperature conversion (if applicable)
        if 'Air temperature [K]' in df.columns:
            df['Air temperature [c]'] = df['Air temperature [K]'] - 273.15
        if 'Process temperature [K]' in df.columns:
            df['Process temperature [c]'] = df['Process temperature [K]'] - 273.15

        # Drop unnecessary columns
        drop_cols = schema.get('drop_columns', [])
        df = drop_columns(df=df, cols=drop_cols)

        # Extract features for transformation
        input_features = df[schema['scaling_features'] + schema['ordinal_columns']]

        # Apply preprocessing
        transformed_features = preprocessor.transform(input_features)
        transformed_df = pd.DataFrame(transformed_features, columns=input_features.columns)

        logging.info("Test data processed successfully.")
        return transformed_df

    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)
