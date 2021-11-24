# Creating API for tarning and peforming predition.
# Date:- 18-NOV-2021

# Importing required libraries. 
from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard

from applicationlogger.setup_logger import setup_logger
from TrainingDataValidation.trainDataValidation import TrainDataValidation
from TrainDataIngestion.trainDataIngestion import TrainDataIngestion
from training import Train
from prediction_data_ingestion.prediction_data_ingestion import PredictionDataIngestion
from prediction_service.prediction import Prediction

# creating logging instance to log API calls
logger = setup_logger("main_log", "api_main.log")

# Importing custom modules.
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Creating flask app variable.
app = Flask(__name__)
dashboard.bind(app)
CORS(app)

# Defifing home rout
@app.route("/", methods=["GET"])
@cross_origin() # helps in deployment time
def home(): # defifing home rout
    logger.info("Home page called.")
    return render_template("index.html") # not in use: not in directory


# Defining training rout
@app.route("/train", methods=["POST"])
@cross_origin() # helps on deployment time
def train():  # defifing training rout
    """
    API for training.
    """
    try:
        logger.info("Training API called.")
        if request.json["batch_folder_path"] is not None: # checking the folder path is not empty or not.
            raw_data_path = request.json["batch_folder_path"] # getting the folder path from the request.
            logger.info("Raw data path: {}".format(raw_data_path))
            # creating object of raw data validation class
            train_data_validation = TrainDataValidation(raw_data_path) # ./Training_Batch_Files/
            # creating object of train data ingestion class
            # train_data_ingestion = TrainDataIngestion() # ./Training_Batch_Files/ # coomented on 18-nov-2021
            logger.info("Raw data ingestion object created.") 

            logger.info("Validating Raw Training data started...")
            train_data_validation.validate_training_data() # validating the training data.
            logger.info("Validating Raw Training data completed.")
            
            logger.info("Data Transformation Started...")
            # train_data_ingestion.data_transformation() # performing data transformation. # commented on 18-nov-2021
            logger.info("Data Transformation Completed.")

            logger.info("Data insertation into database Started...")
            # train_data_ingestion.insert_data_into_database() # inserting data into database. # commented on 18-nov-2021
            # above line shoud be commented for avoiding data insertion in to database.
            logger.info("Data insertation into database Completed.")

            # creating object of train class
            train = Train() # creating object of train class
            logger.info("Train object created.")

            logger.info("Training Data is pulling from database and Preprocessing started...")
            train.preprocess() # pulling data from database and preprocessing.
            logger.info("Training Data is pulling from database and Preprocessing completed.")

            # training the model
            logger.info("Training started...")
            train.train() # training the model.
            logger.info("Training completed.")
            
            logger.info("Training API completed.")
            return Response("Training Succcessfull")
        
    except Exception as e:
        logger.error("Error occured while training: {}".format(e))
        return Response(f"Error occured while training: {e}")


# Defining prediction rout
@app.route("/predict", methods=["POST"])
@cross_origin() # helps on deployment time
def predict():  # defifing prediction rout
    """
    API for prediction.
    """
    try:
        if request.json["batch_folder_path"] is not None:
            raw_data_path = request.json["batch_folder_path"]
            logger.info("Raw data path: {}".format(raw_data_path))
            # creating object of raw data ingestin class
            prediction_data_ingestion = PredictionDataIngestion(raw_data_path) # ./Prediction_Batch_Files/
            
            logger.info("Validating Raw Prediction data started...")
            # prediction_data_ingestion.validate_prediction_data() # validating the prediction data.
            logger.info("Validating Raw Prediction data completed.")

            logger.info("Data Transformation Started...")
            # prediction_data_ingestion.data_transformation() # performing data transformation.
            logger.info("Data Transformation Completed.")

            logger.info("Data insertation into database Started...")
            # prediction_data_ingestion.insert_data_into_database() # inserting data into database.
            logger.info("Data insertation into database Completed.")


            # creating object of prediction class
            pred = Prediction() # creating object of prediction class
            logger.info("Prediction object created.")
            
            logger.info("Prediction Data is pulling from database and Preprocessing started...")
            # pred.predict_data_pull() # pulling data from database and preprocessing.
            logger.info("Prediction Data is pulling from database and Preprocessing completed.")

            # data transformation
            logger.info("Prediction data transformation Started...")
            pred_df = pred.prediction_data_preprocessing() # performing data transformation.
            logger.info("Prediction data transformation Completed.")
            pred.prediction_batch_data(pred_df) # prediction batch data.
            logger.info("Prediction batch data completed.")

            logger.info("Prediction API completed.")
            return Response("Prediction Succcessfull")

    except Exception as e:
        logger.error("Error occured while prediction: {}".format(e))
        return Response(f"Error occured while prediction: {e}")



port = int(os.environ.get("PORT", 5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host, port, app) # creating http server
    httpd.serve_forever() # serving the app.

