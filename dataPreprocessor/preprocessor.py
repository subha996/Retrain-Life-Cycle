# Writing class for basic data preprocessing and making one-hot encoding pipeline. 
# Date: 14-Nov-2021
# Osprey was here...
#--------------------------------------#
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from applicationlogger.setup_logger import setup_logger
from sklearn.impute import KNNImputer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings 
warnings.filterwarnings("ignore") # Ignore warnings


class DataPreprocessor():
    """
    Class for data preprocessing.
    """

    def __init__(self):
        """
        Initialize the class with the data path.
        """
        self.logger = setup_logger("dataPreprocesser_log",
                                    log_file="logs/dataPreprocessor.log")


    # Function to remove columns from the dataframe
    def remove_columns(self, dataframe: DataFrame, columns: list) -> object:
        """
        Method to remove columns from the dataframe.
        Description: This method will be used to remove columns from the dataframe.
        Parameters:
                dataframe: The dataframe to be processed.
                columns: The list of columns to be removed.

        Returns:
                dataframe: The dataframe with the columns removed.
        
        Version: 0.0.1
        """
        try:
            self.logger.info("Removing columns from the dataframe...")
            dataframe.drop(columns, axis=1, inplace=True)
            self.logger.info("Columns removed from the dataframe")
            return dataframe
        except Exception as e:
            print("Error on remove_column method")
            self.logger.error("Error in removing columns from the dataframe: {}".format(e))
            raise e

    
    # Function to separate features and target columns
    def separate_features_target(self, dataframe: DataFrame, target_column: str) ->bool:
        """
        Method to separate features and target columns.
        Description: This method will be used to separate features and target columns from the dataframe.
        Parameters:
                dataframe: The dataframe to be processed.
                target_column: The target column to be separated.
        
        Returns:
                X: The features column.
                y: The target column.
        
        Version: 0.0.1
        """
        try:
            self.logger.info("Separating features and target columns...")
            features = dataframe.drop(target_column, axis=1) # Drop the target column
            target = dataframe[target_column] # Get the target column
            return features, target # Return the features and target columns
        except Exception as e:
            print("Error on separate_features_target method")
            self.logger.error("Error in separating features and target columns: {}".format(e))
            raise e


    # Function to check missing value present
    def is_missing_value(self, data: DataFrame) -> bool:
        """
        Method to check missing value present.
        Description: This method will be used to check missing value present in the dataframe.
        Parameters:
                data: The dataframe to be processed.
        
        Returns:
                True: If missing value is present at any column.
                False: If missing value is not present at any column.
        
        Version: 0.0.1
        """
        try:
            self.logger.info("Checking missing value...")
            return data.isnull().values.any() # Return True if missing value is present at any column
        except Exception as e:
            print("Error on is_missing_value method")
            self.logger.error("Error in checking missing value: {}".format(e))
            raise e





    # Function to impute missing values
    def impute_missing_values(self, dataframe: object) -> object:
        """
        Method to impute missing values.
        Description: This method will be used to impute missing values in the dataframe.
        """
        try:
            self.logger.info("Impute missing values...")
            imputer = KNNImputer(n_neighbors=3)
            data_array = imputer.fit_transform(dataframe) # Impute the missing values
            # converting data_array to dataframe
            dataframe = pd.DataFrame(data_array, columns=dataframe.columns) # Convert the array to dataframe
            self.logger.info("Missing values imputed.")
            return dataframe
        except Exception as e:
            print("Error on impute_missing_values method")
            self.logger.error("Error in imputing missing values: {}".format(e))
            raise e
    

    # Function to get column with zero standard deviation
    def get_column_with_zero_std_dev(self, dataframe: object) -> list:
        """
        Method to get column with zero standard deviation.
        Description: This method will be used to get column with zero standard deviation.
        Parameters:
                dataframe: The dataframe to be processed.
        
        Returns:
            list of columns with zero standard deviation.
            These shall be removed before passing to the model.
        
        Version: 0.0.1
        """
        try:
            self.logger.info("Getting columns with zero standard deviation...")
            std_dev = dataframe.std() # Get the standard deviation of each column
            zero_std_dev_columns = list(std_dev[std_dev == 0].index) # Get the columns with zero standard deviation
            self.logger.info("Columns with zero standard deviation found.")
            return zero_std_dev_columns # return the list of columns with zero standard deviation
        except Exception as e:
            print("Error on get_column_with_zero_std_dev method")
            self.logger.error("Error in getting columns with zero standard deviation: {}".format(e))
            raise e


    # Function to creatin one-hot encoding and scaling pipeline
    def scale_encode_pipeline(self, datafrmae: DataFrame) -> Pipeline:
        """
        Method to create one-hot encoding and scaling pipeline.
        Description: This method will be used to create one-hot encoding and scaling pipeline.
                     One-Hot_Encoding will be used on categorical columns and scaling will be used on numerical columns.
                     This Method shall use on Feature Dataframe(X) only after separating target column.
        Parameters:
                dataframe: The dataframe to be processed.
        
        Returns:
                pipeline: The pipeline for one-hot encoding and scaling.
                
                .fit_transform() will be used on the return object.
        
        Version: 0.0.1
        """
        try:
            self.logger.info("Creating one-hot encoding and scaling pipeline...")
            # Validating input dataframe
            if not isinstance(datafrmae, DataFrame):
                self.logger.error("Invalid input dataframe. It should be a dataframe.")
                raise TypeError("Input dataframe is not of type DataFrame")
            # Getting list of categorical columns and numerical columns list
            cat_cols = datafrmae.select_dtypes(exclude=[np.number]).columns.tolist() # list of categorical columns
            num_cols = datafrmae.select_dtypes(include=[np.number]).columns.tolist() # list of numerical columns names
            # Creating numeric transformer
            numeric_transformer = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])
            # Creating categorical transformer
            categorical_transformer = OneHotEncoder(drop="first",
                                                    sparse=False)

            # Creating final transformer
            column_transformer = ColumnTransformer(transformers=[
                ("num", numeric_transformer, num_cols), # Numerical columns
                ("cat", categorical_transformer, cat_cols) # Categorical columns 
            ])

            self.logger.info("One-hot encoding and scaling pipeline created.")
            return column_transformer # Return the pipeline
        except Exception as e:
            print("Error on scale_encode_pipeline method")
            self.logger.error("Error in creating one-hot encoding and scaling pipeline: {}".format(e))
            raise e



    


        
           
            




