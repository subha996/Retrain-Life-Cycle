# Write class for raw data validation and insertionion of data into database
# Date: 16-Nov-2021
#--------------------------------------#


# importing libraries
import os
from applicationlogger.setup_logger import setup_logger
from data_validation.raw_data_validator import RawDataValidator
from utils.fileoperation import DataGetter

class TrainDataValidation():
    """Class for Validate Training Raw Data.
    This Class Shall call directly from API.
    """


    def __init__(self, train_batch_directory: str):
        """Initialize the logger and data getter class.
        """
        if not os.path.isdir('./Validation_logs'):
            os.mkdir('./Validation_logs') # create directory for logs
        
        self.logger = setup_logger("TrainDataValidation", "Validation_logs/traindatavalidation.log")
        self.utils = DataGetter()
        self.batch_directory = train_batch_directory
        self.train_data_validated = "./Training_Validated_Raw_Files" # path to keep good bad validated files.
        self.validator = RawDataValidator(self.batch_directory,
                                          self.train_data_validated) # pssing batch directory to raw data validator

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
            metadata = self.utils.read_json_file("schema_file\schema_training.json") # jason file path
            # getting data from dictionary
            regex = self.utils.regex_pattern() # return regex pattern for matching file name
            length_date_stamp = metadata["LengthOfDateStampInFile"] # 8
            length_time_stamp = metadata["LengthOfTimeStampInFile"] # 6
            length_column = metadata["NumberofColumns"] # 7
            
            # caliing file validation function
            self.validator.validate_file_name(regex_pattern=regex,
                                                length_date_stamp=length_date_stamp,
                                                length_time_stamp=length_time_stamp)

            self.logger.info("validating training data completed.")

            # validation column length
            self.logger.info("validating training data column length...")
            self.validator.validateColumnLength(length_of_columns=length_column) # passing column length
            self.logger.info("validating training data column length completed.")

            # Validate missing value in whole column
            self.logger.info("validating training data missing value in whole column...")
            self.validator.validateMissingValueWholeColumn()
            self.logger.info("validating training data missing value in whole column completed.")

        except Exception as e:
            self.logger.error(f"Error occured in validate_training_data method: {e}")
            raise e 
        