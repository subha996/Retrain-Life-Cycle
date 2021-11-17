# Writing class for doing prediction from batch data and single observation data.
# Date: 16-Nov-2021
#--------------------------------------#

# importing libraries
import os
import time
from applicationlogger.setup_logger import setup_logger
from data_collector.datapull import DataPull
from utils.fileoperation import DataGetter
from dataPreprocessor.preprocessor import DataPreprocessor


class Prediction():
    """
    This class shall be used for performing prediction on batch data as well as single observation.
    """

    def __init__(self):
        
        self.logger = setup_logger("prediction_log",
                                    "logs/prediction_service.log")
        self.data_pull = DataPull() # creating object for data pull class
        self.utils = DataGetter() # creating object for DataGetter class
        self.preprocessor = DataPreprocessor() # creating object for DataPreprocessor class
        
    # Function to pull data from database and store in local machine
    def predict_data_pull(self):
        """
        This method shall be used to pull data from database and store in local machine.
        Paramater: keyspace : keyspace name on the cloud.
                   table: table name from where data is to be pulled.
        
        Output: None 
                local
        """
        try:
            self.logger.info("Started pulling data from database")
            self.output_path = os.path.join("./Prediction_data_from_DB", "prediction_data.csv") # full path of the file to be stored
            self.data_pull.pull_data(keyspace="insurance1", 
                                    table="predict_data77",
                                    output_file=self.output_path)
            
            self.logger.info("Data pulled from database and stored in local machine")
            # sleeping for 5 seconds to wait for data to be stored in local machine
            time.sleep(5)
        
        except Exception as e:
            self.logger.error("Error occured while pulling data from database" + str(e))
            raise Exception("Error occured while pulling data from database" + str(e))            
            

    # Function to perform data preprocessing on the data
    def prediction_data_preprocessing(self):
        """
        This method shall be used to perform data preprocessing on the data
        Parameter: None
        Output: None

        Version: 0.0.1

        """
        try:
            self.logger.info("Started preprocessing data...")
            # getting the data form local directory
            pred_data = self.utils.read_csv_file(self.output_path) # return dataframe

            # preprocessing the data
            self.logger.info("Preprocessing data...")
            # checking null values
            is_missing = self.preprocessor.is_missing_value(pred_data) # return boolean value
            if is_missing:
                self.preprocessor.impute_missing_values(pred_data) # impute missing values
                self.logger.info("Missing values imputed")
            else:
                self.logger.info("No missing values found")
            e





            


    
    
    
    
    
    
    
    def prediction_on_batch_data(self):
        
        """
        This method shall be used to perform predicton on batch data
        """

    

