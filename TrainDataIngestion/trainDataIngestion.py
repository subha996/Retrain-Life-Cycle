# Write class for raw data validation and insertionion of data into database
# Date: 16-Nov-2021
#--------------------------------------#


# importing libraries
import os
from applicationlogger.setup_logger import setup_logger
from data_transformation.dataTransformation import DataTransformation
from dataingestion.cassandra_helper import CassandraHelper
from utils.fileoperation import DataGetter


class TrainDataIngestion():
    """class for sending data into Database
    """
    def __init__(self):

        # setting up logger
        if not os.path.isdir("./dataBaselogs"):
            os.mkdir("./dataBaselogs") # create directory if not exists
        self.logger = setup_logger("TrainDataIngestion", 
                                    log_file="dataBaselogs/train_data_validation_ingestion.log")
        self.train_data_validated = "./Training_Validated_Raw_Files" # path to keep good bad validated files.
        self.data_transformer = DataTransformation() # data transformation class
        self.utils = DataGetter() # file operation class
        self.database = CassandraHelper() # cassandra helper class for database connection



    def data_transformation(self):
        """
        This method shall be used before dump data into database.
        Parameter: None
        Output: None

        Version: 0.0.1
        """
        try:
            self.logger.info("Data transformation started...")
            # getting the path of good file to be transformed
            good_files = os.path.join(self.train_data_validated, "good") # path to good files

            # calling ddata transformation methdod to replace nan with NULL
            self.data_transformer.replace_null_with_NULL(good_files)
            self.logger.info("Data transformation completed.")
        
        except Exception as e:
            self.logger.error(f"Error occured in data_transformation method: {e}")
            raise e

    
    # Function to send transformed data to database
    def insert_data_into_database(self):
        """
        This method shall used to insert data into cassandra database 
        Note: Use this method after validating and transformation of the raw data.

        Parameter:
                None
        Output: None

        Version: 0.0.1
        """
        try:
            self.logger.info("Inserting data into database started...")
            good_files = os.path.join(self.train_data_validated, "good") # path to good files
            # getting list of files from good files
            file_list = os.listdir(good_files)
            print(file_list)
            # looping through all files
            for csv_file in file_list:
                # getting full path of file
                full_path = os.path.join(good_files, csv_file)
                # calling cassandra helper class to insert data into database
                self.database.send_csv(file=full_path, # passing csv file
                                        keyspace="insurance1", # NOTE: keyspace shoud be created manually
                                                             # NOTE: `send_csv` assumed keyspace is already created. 
                                        table_name="train_data77" # NOTE: table will created automatically if not exists.
                                        )
                print(f"{csv_file} inserted into database.")
                self.logger.info(f"Inserting data into database completed for file: {csv_file}")
            self.logger.info("Inserting data into database completed.")
        
        except Exception as e:
            self.logger.error(f"Error occured in insert_data_into_database method: {e}")
            raise e


        