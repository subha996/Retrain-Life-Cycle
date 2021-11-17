# Write class for raw data validation and insertionion of data into database
# Date: 16-Nov-2021
#--------------------------------------#

# importing libraries
import os
from applicationlogger.setup_logger import setup_logger
from data_validation.raw_data_validator import RawDataValidator
from data_transformation.dataTransformation import DataTransformation
from dataingestion.cassandra_helper import CassandraHelper
from utils.fileoperation import DataGetter



class TrainDataIngestion():
    """
    Class for data validation and Ingestion into database.
    """
    def __init__(self, predict_batch_directory: str):

        # setting up logger
        self.logger = setup_logger("prediction_data_validation and ingestion", 
                                    log_file="logs/prediction_data_validation_ingestion.log")
        self.batch_directory = predict_batch_directory
        self.predict_data_validated = "./Prediction_Validated_Raw_Files" # path to keep good bad validated files.
        self.validator = RawDataValidator(self.batch_directory,
                                          self.predict_data_validated) # pssing batch directory to raw data validator
        self.data_transformer = DataTransformation() # data transformation class
        self.utils = DataGetter() # file operation class
        self.database = CassandraHelper() # cassandra helper class for database connection


    def validate_training_data(self):
        """
        This function will validate the raw data and write the good and bad files into respective folders.
        Parameter: 
            None
        Return:
            None

        Version: 0.0.1
        """
        try:
            self.logger.info("validating training data started...")
            
            # reading metadat from  json file
            metadata = self.utils.read_json_file("schema_file\schema_prediction.json") # jason file path
            # getting data from dictionary
            regex = self.utils.regex_pattern() # return regex pattern for matching file name
            length_date_stamp = metadata["LengthOfDateStampInFile"] # 8
            length_time_stamp = metadata["LengthOfTimeStampInFile"] # 6
            length_column = metadata["NumberofColumns"] # 6, : 1 less from traning data as target is missing.
            
            # caliing file validation function
            self.validator.validate_file_name(regex_pattern=regex,
                                               length_date_stamp=length_date_stamp,
                                               length_time_stamp=length_time_stamp)

            self.logger.info("validating prediction data completed.")

            # validation column length
            self.logger.info("validating prediction data column length...")
            self.validator.validateColumnLength(length_of_columns=length_column) # passing column length
            self.logger.info("validating prediction data column length completed.")

            # Validate missing value in whole column
            self.logger.info("validating prediction data missing value in whole column...")
            self.validator.validateMissingValueWholeColumn()
            self.logger.info("validating prediction data missing value in whole column completed.")

        except Exception as e:
            self.logger.error(f"Error occured in validate_prediction_data method: {e}")
            raise e
    
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
            good_files = os.path.join(self.predict_data_validated, "good")

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
            good_files = os.path.join(self.predict_data_validated, "good") # path to good files
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
                                        table_name="predict_data77" # NOTE: table will created automatically if not exists.
                                        )
                print(f"{csv_file} inserted into database.")
                self.logger.info(f"Inserting data into database completed for file: {csv_file}")
            self.logger.info("Inserting data into database completed.")
        
        except Exception as e:
            self.logger.error(f"Error occured in insert_data_into_database method: {e}")
            raise e
            
                
     


    

        




        