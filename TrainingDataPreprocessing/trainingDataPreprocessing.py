# Write class for preprocessing traning data 
# Date: 16-Nov-2021
# --------------------------------------#

# Importing libraries
import os
import time
import pandas as pd
from applicationlogger.setup_logger import setup_logger
from utils.fileoperation import DataGetter
from data_collector.datapull import DataPull
from dataPreprocessor.preprocessor import DataPreprocessor


class TrainingDataPreprocessing():
    def __init__(self):
        self.logger = setup_logger("trainingDataPreprocessing", "logs/trainingDataPreprocessing.log")
        self.utils = DataGetter() # creating object for DataGetter class
        self.dataPull = DataPull() # creating object for DataPull class
        self.dataPreprocessor = DataPreprocessor() # creating object for DataPreprocessor class


    # Functoion for preprocessing training data
    def preprocess_training_data(self):
        """
        This method will use to preprocess the tarning data and will return the dataframe ready to Feed to the algorithm.
        Parameter: None
        Output: Saved Preprocess data into local directory aftere preprocessing..

        On Error: log error

        Versin: 0.0.1
        """
        try:
            self.logger.info("Preprocessing training data started...")
            # getting the data from database
            train_file_path = os.path.join("Training_Data_From_DB", "train_insurance.csv")
            self.dataPull.pull_data(keyspace="insurance",
                                    table="insurance_data",
                                    output_file=train_file_path) # return dataframe.

            # sleepin for 5 seconds
            time.sleep(5) # it may take some time to pull data from database. and save it to csv file. 

            # Reading the data from local directory.
            data = self.utils.read_csv_file(train_file_path) # return dataframe.

            # imputing missing value if present
            self.logger.info("Imputing missing values if present...")
            if self.dataPreprocessor.is_missing_value(data): # check if missing value is present or not.
                data = self.dataPreprocessor.impute_missing_values(data) # if present impute the missing value.
                self.logger.info("Missing values imputed.")
            else:
                self.logger.info("No missing values has been found.")

            # getting column with zero standard deviation
            self.logger.info("Getting column with zero standard deviation...")
            col_list_zero_std = self.dataPreprocessor.get_column_with_zero_std_dev(data) # return list of column with zero standard deviation.
            self.logger.info("Column with zero standard deviation: {}".format(col_list_zero_std))
            # removing the column with zero standard deviatio
            if len(col_list_zero_std) > 0:
                data = self.dataPreprocessor.remove_columns(data, col_list_zero_std) # removing the column with zero standard deviation.
                self.logger.info("Columns with zero standard deviation removed.")
            else:
                self.logger.info("No columns with zero standard deviation found.")
            
            # Saving the dataframe to csv file for future use.
            preprocess_file_path = os.path.join("Training_Data_Preprocessed", "train_insurance_preprocessed.csv")
            self.utils.write_csv_file(data, 
                                      preprocess_file_path) # saving the dataframe to csv file.
            self.logger.info("Preprocessing training data completed.")

        except Exception as e:
            self.logger.error("Error while preprocessing training data. Error: {}".format(e))
            raise e
    

    def data_scaling_encoding(self):
        """
        This method will be uses to create pipeline for data scaling and encoding
        Parameter: None
        Output: Saved pipeline in local directory,  
                Return dataframe after column transformation.

        On Error: log error
        
        Verson: 0.0.1
        """
        try:
            self.logger.info("Saving pipeline for data scaling and encoding Started...")
            # getting the data from local
            data = self.utils.read_csv_file(os.path.join("Training_Data_Preprocessed", 
                                                        "train_insurance_preprocessed.csv")) # return dataframe.
            
            # separate the target column from the rest of the columns
            X, y = self.dataPreprocessor.separate_features_target(data,
                                                                "expenses") # return dataframe.
            
            # scaling the numerical data and encoding the categorical data
            column_transformer = self.dataPreprocessor.scale_encode_pipeline(X) # return column transformer.
            # Fitting the column transformer into data
            X = column_transformer.fit_transform(X) # return array.

            # converting it to dataframe
            X = pd.DataFrame(X) # Note this datafrmae column names will be changed. as 0, 1, 2 so on.
            
            # savinng the pipeline in local directory
            file_directory = "./ColumnTransformer" # saving the pipeline in local directory.
            os.makedirs(file_directory, exist_ok=True) # creating directory if not present.
            file_path = os.path.join(file_directory, "column_transformer") # saving the pipeline in local directory.
            self.utils.write_pickle_file(column_transformer,
                                        file_path=file_path) # saving the pipeline in local directory.
            self.logger.info("Saving pipeline for data scaling and encoding Completed.")
            
            X["expenses"] = y # adding the target column to the dataframe.
            return X # return the dataframe.
        
        except Exception as e:
            self.logger.error("Error while saving pipeline for data scaling and encoding. Error: {}".format(e))
            raise e



