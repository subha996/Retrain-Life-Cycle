# This is the main gateway for the entire application.

# Imporing alll module
from training_data_ingestion.train_data_ingestion import TrainDataIngestion
from training import Train
# ----------------------------PART 1-------------------------------#
# Performing data validation and sending data to the database.
# creating object of traindataingestion class
data_ingestion = TrainDataIngestion("./Training_Batch_Files") # passing the raw data path of the folder

data_ingestion.validate_training_data() # for validatin file name.
data_ingestion.data_transformation() # for data transformation. replace nan with NULL
data_ingestion.insert_data_into_database() # for inserting data into database.'
# data sent to the database.
#--------------------------Data Ingestion End--------------------------------#


# #----------------------------PART 2-------------------------------#
# #-----------------------------------------TRAINING-----------------------------------------#
# # creating class for Train class
train = Train()

train.preprocess() # pulling data from database and preprocessing.
train.train()  # for training the model. and saving the model.
#----------------------------------Train End---Model Saved---------------------------------#


# # #----------------------------PART 3-------------------------------#
# # #--------------------------PREDICTION--------------------------------#
from prediction_data_ingestion.prediction_data_ingestion import PredictionDataIngestion
from prediction_service.prediction import Prediction

# # creating object of prediction data ingestion class
pred_data_inge = PredictionDataIngestion("./Prediction_Batch_Files") # passing the raw data path of the folder

pred_data_inge.validate_prediction_data() # for validatin file name.
pred_data_inge.data_transformation() # for data transformation. replace nan with NULL
pred_data_inge.insert_data_into_database() # for inserting data into database.:
# data sent to the database.
# #---------------------------Data Ingestion End--------------------------------#

# # Creating object of prediction service class
pred = Prediction() 
pred.predict_data_pull() # Pulling data form database.
pred_data_encoded = pred.prediction_data_preprocessing() # preprocessing the data. return dataframe.
pred.prediction_batch_data(pred_data_encoded) # doing prediction on the data on encoded data.
# # above method will store predicton output file in to "./Prediction_Output_File" path







