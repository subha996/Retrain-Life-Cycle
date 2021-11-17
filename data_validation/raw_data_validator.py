# Writing class for raw data validation
# Date: 30-Oct-2021
# osprey
#--------------------------------------#

# Imorting necessary libraries
import os
import re
import shutil
from utils.fileoperation import DataGetter
from applicationlogger.setup_logger import setup_logger
import warnings
warnings.filterwarnings("ignore") # to ignore warnings


class RawDataValidator():
    """docstring later
    """
    def __init__(self, batch_directory: str, validated_raw_directory: str):
        self.batch_directory = batch_directory
        # Setting up logger
        self.log = setup_logger("RawDataValidation_logs", "logs/Raw_data_validator.log")
        self.log.info("-"*30 + "Season Divider" + "-"*30) # seperating logs for different seasons
        self.utils = DataGetter() # Creating object for utils class
        # defining name of validated good and bad folder
        self.validated_raw = validated_raw_directory
        self.good_raw = os.path.join(self.validated_raw, "good") # full path to good folder
        self.bad_raw = os.path.join(self.validated_raw, "bad") # full path to bad folder.
    
    # Function to validate raw input data
    def validate_file_name(self, 
                                regex_pattern: str,
                                length_date_stamp: int,
                                length_time_stamp: int) -> None:
        """
        Method: validate_raw_input_data
        Description : This method validates raw input data from bathc directory
        Parameters : batch_directory (str)
        Return : None

        Version: 0.0.1
        """
        try:
            self.log.info("Entered validate_raw_input_data method...")

            # Creating new folder for storing validated files
            self.log.info("Creating new folder for storing validated files...")
            self.utils.create_directory(
                                        directory_path = self.validated_raw # directory path for good bad files
                                        ) # creating directory for storing va;idated files
            self.log.info("New folder created successfully.")

            # Getting list of files from batch directory
            file_list = os.listdir(self.batch_directory) # getting list of files from batch directory
            
            # Looping through all file to validate.
            for file_name in file_list:
                # Matching the file name with regex 
                if (re.match(regex_pattern, file_name)):
                    split_at_dot = re.split(".csv", file_name) # splitting file name on dot
                    split_at_dot = re.split("_", split_at_dot[0]) # splitting file name on underscore
                    if len(split_at_dot[1]) == length_date_stamp: # Checking the date stamp length
                        if len(split_at_dot[2]) == length_time_stamp: # Checking the time stamp length
                            self.utils.move_file(source_directory=self.batch_directory,
                                                    destination_directory = self.good_raw,
                                                    file_name=file_name) # moving file to validated files directory
                            self.log.info(f"File {file_name} moved to good folder.")
                        else: # else time_stamp length is not matching
                            self.log.info(f"File {file_name} having incorrect Time Stamp length, moving to bad folder.")
                            self.utils.move_file(source_directory=self.batch_directory,
                                                    destination_directory = self.bad_raw,
                                                    file_name=file_name) # moving file to validated files directory

                    else: # else date_stamp length is not matching
                        self.log.info(f"File {file_name} having incorrect Date Stamp length, moving to bad folder.")
                        self.utils.move_file(source_directory=self.batch_directory,
                                                destination_directory = self.bad_raw,
                                                file_name=file_name) # moving file to validated files directory

                else: # else file name is not matching with regex
                    self.log.info(f"File {file_name} having incorrect file name, moving to bad folder.")
                    self.utils.move_file(source_directory = self.batch_directory,
                                            destination_directory = self.bad_raw,
                                            file_name = file_name) # moving file to validated files directory
            self.log.info("Validated files moved successfully.")

        except Exception as e:
            self.log.error(f"Error occured in validate_raw_input_data method: {e}")
            raise e
        finally:
            self.log.info("Exited validate_raw_input_data method...")

    
    # Function to validatye column length
    def validateColumnLength(self, length_of_columns):
        """
        Method: validateColumnLength
        Description : This method validates column length of raw input data
        Parameters : None
        Return : None

        Version: 0.0.1
        """
        try:
            self.log.info("Entered validateColumnLength method...")

            # Getting list of files from batch directory
            file_list = os.listdir(self.good_raw) # getting list of files from batch directory
            
            # Looping through all file to validate.
            for file_name in file_list:
                full_path = os.path.join(self.good_raw, file_name) # full path of file
                # Getting file data
                file_data = self.utils.read_csv_file(full_path) # getting file data
                # Getting number of columns
                number_of_columns_in_file = len(file_data.columns) # getting number of columns
                if number_of_columns_in_file == length_of_columns: # Checking number of columns
                    self.log.info(f"File {file_name} having correct number of columns, keeping it on good folder.")
                    pass  # keeping file on good folder as it has correct number of columns
                else: # else number of columns is not matching
                    self.utils.move_file(source_directory  = self.good_raw,
                                            destination_directory = self.bad_raw,
                                            file_name = file_name)
                    self.log.info(f"File {file_name} having incorrect number of columns, moved to bad folder.")
            self.log.info("Validated files moved successfully.")
        except Exception as e:
            self.log.error(f"Error occured in validateColumnLength method: {e}")
            raise e
            

    
    # Function to validate missing value in whole column.
    def validateMissingValueWholeColumn(self):
        """
        Method: validateMissingValueWholeColumn
        Description : This method validates missing value in whole column.
                        If such column found that have whole column have 
                        missing value then it moved to bad folder.
        Parameters : data_directory (str)
        Return : None

        Version: 0.0.1
        """
        try:
            self.log.info("Entered validateMissingValueWholeColumn method...")
            file_list = os.listdir(self.good_raw) # getting list of files from data directory
            
            # Looping through all file to validate.
            for csv_file in file_list:
                # csv file full path
                full_path = os.path.join(self.good_raw, csv_file)
                # Reading file
                df = self.utils.read_csv_file(full_path) # reading csv file return as dataframe.
                # taking a variable to count columns which have whole columns missing value.
                count_columns_with_missing_value = 0 # initial it will be zero (0)
                
                # Looping through all columns
                for col in df:
                    if (len(df[col]) - df[col].count()) == len(df[col]): # checking if column have missing value
                        count_columns_with_missing_value += 1 # if column have missing value then incrementing count
                        shutil.move(full_path, self.bad_raw) # moving file to validated files directory
                        self.log.info(f"File {csv_file} having missing value in whole {col} :column, moving to bad folder.")
                        break # if one such column found it will break.
                              # because we have to move file to bad folder.
                    else:
                        continue # if no such column found then it will continue.
                
                if count_columns_with_missing_value == 0: # if count is zero then file is valid
                    self.log.info(f"File {csv_file} having no missing value in whole column, keeping in good folder.")
                    pass # file already on good folder.
            self.log.info("Validated missing value in whole column successfully.")
        
        except Exception as e:
            self.log.error(f"Error occured in validateMissingValueWholeColumn method: {e}")
            raise e
        finally:
            self.log.info("Exited validateMissingValueWholeColumn method.")