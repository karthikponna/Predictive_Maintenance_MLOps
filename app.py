import sys, os
import pandas as pd

import certifi
from dotenv import load_dotenv

import pymongo

from machine_predictive_maintenance.exception.exception import MachinePredictiveMaintenanceException
from machine_predictive_maintenance.logging.logger import logging
from machine_predictive_maintenance.pipeline.training_pipeline import TrainingPipeline

from machine_predictive_maintenance.utils.main_utils.utils import load_object, processing_test_data
from machine_predictive_maintenance.utils.ml_utils.model.estimator import MachinePredictiveModel

from machine_predictive_maintenance.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME
from machine_predictive_maintenance.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from machine_predictive_maintenance.constant.training_pipeline import SCHEMA_FILE_PATH

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse


ca = certifi.where()
load_dotenv()

mongo_db_url = os.getenv("MONGO_DB_URL")

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    
    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile=File(...)):
    try:
        df=pd.read_csv(file.file)
        #print(df)
        preprocesor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")

        processing_data = processing_test_data(data= df, schema_file= SCHEMA_FILE_PATH, preprocessor= preprocesor)

        predictive_model = MachinePredictiveModel(model= final_model)
        print(processing_data.iloc[0])
        
        y_pred = predictive_model.predict(processing_data)
        print(y_pred)
        
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)

        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    
    except Exception as e:
        raise MachinePredictiveMaintenanceException(e, sys)


    
if __name__== "__main__":
    app_run(app, host="0.0.0.0", port = 8080)
    