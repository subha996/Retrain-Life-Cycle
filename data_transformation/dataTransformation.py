# Writng class to data transofrmation: before insert data to database
# Date: 14-Nov-2021
# Osprey was here...

import os
from applicationlogger.setup_logger import setup_logger
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore") # to ignore the warning messages


class DataTransformation():
    """
    Data Transformation: This class shall be used to perform data transformation  before inserting data to database.
    Methods: replace_null_with_NULL

    Version: 0.0.1
    """
    def __init__(self):
        self.log = setup_logger("file_operation_log", "logs/file_operation.log")

    def replace_null_with_NULL(self, raw_batch_folder_path) -> None:
        """
        Method: replace_null_with_NULL
        Description: This method is used to replace null values with NULL in the dataframe
        Parameters:
            raw_batch_folder_path: path of the raw batch folder: str
        Return: None
        
        Version: 0.0.1
        """
        try:
            # Validating the file path its string type
            if not isinstance(raw_batch_folder_path, str):
                raise TypeError("File path Must be String(str) type object.") # raising error if file path is not string type
            # Checking the path exists or not 
            if os.path.exists(raw_batch_folder_path):
                # Getting the list of files from the raw batch folder
                raw_batch_files = os.listdir(raw_batch_folder_path)
                # Looping through the list of files
                for file in raw_batch_files:
                    # Reading the file
                    data = pd.read_csv(raw_batch_folder_path + file, low_memory=False)
                    # Replacing null values with NULL
                    data = data.fillna("NULL") # replacing null values with NULL
                    # Writing the dataframe to the file into same file path.
                    data.to_csv(raw_batch_folder_path + file, index=False)
            else:
                self.log.error("File path is not found or not valid.")
                raise FileNotFoundError("File path is not found or not valid.")
        except Exception as e:
            print("Error in replacing null values with NULL: {}".format(e))
            self.log.error("Error in replacing null values with NULL: {}".format(e))
            raise e

    # Function to replace NULL with np.nan
    def replace_NULL_with_np_nan(self, data) -> None:
        """
        Method: replace_NULL_with_np_nan
        Description: This method is used to replace NULL values with np.nan in the dataframe
                     This method should be call after pulling data from Database.
        Parameters:
            raw_batch_folder_path: path of the raw batch folder: str
        Return: None
        
        Version: 0.0.1
        """
        try:
            # Validating the file path its string type
            if not isinstance(data, pd.DataFrame):
                raise TypeError("File path Must be String(str) type object.") # raising error if file path is not string type
            
            # replacing null values with np.nan
            data = data.replace("NULL", np.nan) # replacing null values with np.nan
            return data # returning the dataframe
            
        except Exception as e:
            print("Error in replacing NULL with np.nan: {}".format(e))
            self.log.error("Error in replacing NULL with np.nan: {}".format(e))
            raise e
