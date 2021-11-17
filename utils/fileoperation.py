# Writing class to read file form local directory
# Date: 30-Oct-2021
# Osprey
#------------------------------------------------#

# Importing necessary library
import os
import pandas as pd
import json
from pandas.core.frame import DataFrame
import yaml
import shutil
import joblib
from applicationlogger.setup_logger import setup_logger
import warnings
warnings.filterwarnings("ignore") # to ignore the warning messages

class DataGetter():
    def __init__(self):
        self.log = setup_logger("file_operation_log", "logs/file_operation.log")


    # Function to read json file form local directory:
    def read_json_file(self, file_path) -> dict:
        """
        Method: read_json_file
        Description: This method is used to read json file from local directory
        Parameters:
            file_path: path of the file: str
        Return: dictionary

        Version: 0.0.1
        """
        try:
            # Validating the file path its string type
            if not isinstance(file_path, str):
                raise TypeError("File path Must be String(str) type object.") # raising error if file path is not string type
            # Checking the path exists or not 
            if os.path.exists(file_path):
                with open(file_path, "r") as json_file: # opening the file
                    data = json.load(json_file) # reading the json file and storing in data
                    json_file.close() # closing the file
                    return data # returning the data as dictionary
            else:
                self.log.error("File path is not found or not valid.")
                raise FileNotFoundError("File path is not found or not valid.")
        except Exception as e:
            print("Error in reading json file: {}".format(e))
            self.log.error("Error in reading json file: {}".format(e))
            raise e

    
    # Function to read yaml file form local directory:
    def read_yaml_file(self, file_path) -> dict:
        """
        Method: read_yaml_file
        Description: This method is used to read yaml file from local directory
        Parameters:
            file_path: path of the file: str
        Return: dictionary

        Version: 0.0.1
        """
        try:
            # Validating the file path its string type
            if not isinstance(file_path, str):
                raise TypeError("File path Must be String(str) type object.") # raising error if file path is not string type
            # Checking the path exists or not
            if os.path.exists(file_path):
                with open(file_path, "r") as yaml_file: # opening the file
                    data = yaml.safe_load(yaml_file) # reading the yaml file and storing in data
                    yaml_file.close() # closing the file
                    return data # returning the data as dictionary
            else:
                self.log.error("File path is not found or not valid.")
                raise FileNotFoundError("File path is not found or not valid.")
        except yaml.YAMLError as e:
            print("Error in reading yaml file: {}".format(e))
            self.log.error("Error in reading yaml file: {}".format(e))
            raise e

    
    # Function to read csv file form local directory:
    def read_csv_file(self, file_path) -> pd.DataFrame:
        """
        Method: read_csv_file
        Description: This method is used to read csv file from local directory
        Parameters:
            file_path: path of the file: str
        Return: pandas dataframe

        Version: 0.0.1
        """
        try:
            # Validating the file path its string type
            if not isinstance(file_path, str):
                raise TypeError("File path Must be String(str) type object.") # raising error if file path is not string type
            # Checking the path exists or not
            if os.path.exists(file_path):
                data = pd.read_csv(file_path) # reading the csv file and storing in data
                return data # returning the data as pandas dataframe
            else:
                self.log.error("File path is not found or not valid.")
                raise FileNotFoundError("File path is not found or not valid.")
        except Exception as e:
            print("Error in reading csv file: {}".format(e))
            self.log.error("Error in reading csv file: {}".format(e))
            raise e


    # Function to write csv file in local directory
    def write_csv_file(self, dataframe: DataFrame, file_path: str):
        """
        Method: write csv file into local directory
        Description: This method shall be use to write dataframe object as cvs file in local directiry.
        
        Parameter: dataframe: Dataframe that to saved
                     file_path: path of the file: str

        Output: None

        Version: 0.0.1
        """
        try:
            # Validating the dataframe object
            if not isinstance(dataframe, DataFrame):
                raise TypeError("Dataframe object Must be DataFrame type object.") # raising error if dataframe is not dataframe type
            # validating the file path
            if not isinstance(file_path, str):
                raise TypeError("File path Must be String(str) type object.")
            # saving the dataframe object as csv file
            dataframe.to_csv(file_path, index=False) # writing the dataframe object into csv file
        except Exception as e:
            print("Error in writing csv file: {}".format(e))
            self.log.error("Error in writing csv file: {}".format(e))
            raise e
    

    # Move to validation module class.
    # Function to return regex pattern to validate file name
    def regex_pattern(self) -> str:
        """
        Method: regex_pattern
        Description: This method is used to return regex pattern to validate file name
        Parameters:
            None
        Return: regex pattern as string

        Version: 0.0.1
        """
        regex_pattern = "['insurance']+['\_'']+[\d_]+[\d]+\.csv" # regex pattern to validate file name
        return regex_pattern # returning the regex pattern

    
    # Function to create good and bad directory to store good file and bad file.
    def create_directory(self, directory_path) -> None:
        """
        Method: create_directory
        Description: This method is used to create good and bad directory to store good file and bad file.
        Parameters:
            directory_path: path of the directory: str
        Return: None

        Version: 0.0.1
        """
        try:
            # Validating the directory path its string type
            if not isinstance(directory_path, str):
                raise TypeError("Directory path Must be String(str) type object.") # raising error if directory path is not string type
            self.log.info(f"Creating good bad folder...")
            # Checking the  good folder path exists or not
            good_path = os.path.join(directory_path + "/", "good/") # creating good path
            if not os.path.isdir(good_path):
                os.makedirs(good_path) # creating the directory
                self.log.info(f"good folder created")
            # Checking bad folder path exists or not
            bad_path = os.path.join(directory_path + "/", "bad/") # creating bad path
            if not os.path.isdir(bad_path):
                os.makedirs(bad_path) # creating the directory
                self.log.info(f"bad folder is created.")
                
            
        except Exception as e:
            print("Error in creating directory: {}".format(e))
            self.log.error("Error in creating directory: {}".format(e))
            raise e

        
    # Function to move file to one directory to another directory
    def move_file(self, source_directory, destination_directory, file_name) -> None:
        """
        Method: move_file
        Description: This method is used to move file to one directory to another directory
        Parameters:
            source_directory: path of the source directory: str
            destination_directory: path of the destination directory: str
            file_name: name of the file: str
        Return: None

        Version: 0.0.1
        """
        try:
            # Validating the source directory path its string type
            if not isinstance(source_directory, str):
                raise TypeError("Source directory path Must be String(str) type object.") # raising error if source directory path is not string type
            # Validating the destination directory path its string type
            if not isinstance(destination_directory, str):
                raise TypeError("Destination directory path Must be String(str) type object.") # raising error if destination directory path is not string type
            # Validating the file name its string type
            if not isinstance(file_name, str):
                raise TypeError("File name Must be String(str) type object.") # raising error if file name is not string type
            # Checking the source directory path exists or not
            if os.path.exists(source_directory):
                # Checking the destination directory path exists or not
                if os.path.exists(destination_directory):
                    # Checking the file name exists or not
                    if os.path.exists(source_directory + "/" + file_name):
                        shutil.copy(source_directory + "/" + file_name, destination_directory) # moving the file to destination directory
                    else:
                        self.log.error("File name is not found or not valid.")
                else:
                    self.log.error("Destination directory path is not found or not valid.")
            else:
                self.log.error("Source directory path is not found or not valid.")
        
        except Exception as e:
            print("Error in moving file: {}".format(e))
            self.log.error("Error in moving file: {}".format(e))
            raise e

    # Function to save model in local directory
    def write_pickle_file(self, model: object, file_path: str, cluster = None) -> None:
        """
        This function should be used to saved model in local directory as pickle file.
        """
        try:
            # Validating the model object
            if not isinstance(model, object):
                raise TypeError("Model object Must be object type object.") # raising error if model is not object type
            # Validating the file path
            if not isinstance(file_path, str):
                raise TypeError("File path Must be String(str) type object.")
            # Saving the model object as pickle file
            if cluster is not None: # if cluster number is provied
                file_path = file_path + "_" + str(cluster) + ".pkl" # creating the file path with cluster number
                joblib.dump(model, file_path) # saving the model object as pickle file
            else: # if cluster number is not provied
                file_path = file_path + ".pkl" # saving the model object as pickle file
                joblib.dump(model, file_path) # saving the model object as pickle file
        
        except Exception as e:
            print("Error in saving model: {}".format(e))
            self.log.error("Error in saving model: {}".format(e))
            raise e
    

    # Function to write yaml file to local directory
    def write_yaml_file(self, file_path, data) -> None:
        """
        Method: write_yaml_file
        Description: This method is used to write yaml file to local directory
        Parameters:
            file_path: path of the file: str
            data: data to be written in file: dict
        Return: None

        Version: 0.0.1
        """
        try:
            # Validating the file path its string type
            if not isinstance(file_path, str):
                raise TypeError("File path Must be String(str) type object.") # raising error if file path is not string type
            
            # Validating the data its dictionary type
            if not isinstance(data, dict):
                raise TypeError("Data Must be Dictionary(dict) type object.")
            
            # dumping data into yaml file.
            with open(file_path, "w") as yaml_file: # opening the file
                yaml.dump(data, yaml_file) # writing the data in file
                yaml_file.close() # closing the file

        except Exception as e:
            print("Error in writing yaml file: {}".format(e))
            self.log.error("Error in writing yaml file: {}".format(e))
            raise e
    

    
