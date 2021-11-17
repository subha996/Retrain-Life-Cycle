# Writing class for doing prediction from batch data and single observation data.
# Date: 16-Nov-2021
#--------------------------------------#

# importing libraries
import os
import re
import time
from pandas.core.frame import DataFrame
from applicationlogger.setup_logger import setup_logger
from data_collector.datapull import DataPull
from utils.fileoperation import DataGetter
from dataPreprocessor.preprocessor import DataPreprocessor
import pandas as pd
import joblib
import itertools

class Prediction():
    """
    This class shall be used for performing prediction on batch data as well as single observation.
    """

    def __init__(self):
        
        self.logger = setup_logger("prediction_log",
                                    "logs/prediction_service.log")
        self.logger.info("Prediction class initialized" + "*"*50)
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
            output_path = os.path.join("./Prediction_data_from_DB", "prediction_data.csv") # full path of the file to be stored
            self.data_pull.pull_data(keyspace="insurance1", 
                                    table="predict_data77",
                                    output_file=output_path)
            
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
            output_path = os.path.join("./Prediction_data_from_DB", "prediction_data.csv")
            self.pred_data = self.utils.read_csv_file(output_path) # return dataframe

            # preprocessing the data
            self.logger.info("Preprocessing data...")
            # checking null values
            is_missing = self.preprocessor.is_missing_value(self.pred_data) # return boolean value
            if is_missing: # if miising value found in the data 
                 pred_data = self.preprocessor.impute_missing_values(self.pred_data) # impute missing values
                 self.logger.info("Missing values imputed")
            else:
                self.logger.info("No missing values found in the prediction dataset.")

            # Checking column with zero standard deviation
            zero_std_columns = self.preprocessor.get_column_with_zero_std_dev(self.pred_data) # return list of columns
            if len(zero_std_columns) > 0: # if zero standard deviation found in the data
                self.logger.info("Columns with zero standard deviation found in the data")
                self.logger.info(f"Shape of data before droping columns: {self.pred_data.shape}")
                pred_data = pred_data.drop(zero_std_columns, axis=1) # drop the columns
                self.logger.info("Zero standard deviation columns dropped")
                self.logger.info(f"Shape of data after droping columns: {self.pred_data.shape}")
            else:
                self.logger.info("No columns with zero standard deviation found in the data")
            
            # Scaling and encoding the data.
            self.logger.info("Scaling and encoding data...")
            # getting the columntransformer pipeline
            column_transformer = joblib.load("ColumnTransformer\column_transformer.pkl") # return pipeline
            pred_array = column_transformer.transform(self.pred_data) # return array
            self.logger.info("Data scaled and encoded")

            # Converting array to dataframe
            pred_encoded_df = pd.DataFrame(pred_array) # return dataframe. The datafame shape hass been changed.
            self.logger.info("Array converted to dataframe Returning dataframe.")

            # Returning data frame
            return pred_encoded_df
        
        except Exception as e:
            self.logger.error("Error occured while preprocessing data" + str(e))
            raise Exception("Error occured while preprocessing data" + str(e))

        
    # Function to perform prediction on batch data
    def prediction_batch_data(self, data: DataFrame, keep_features: bool = False) -> None:
        """
        Method: prediction_batch_data
        Description: This method shall use to perform batch data prediction.

        Parameter:
                data: DataFrame: dataframe to be used for prediction. 
                      It should be encoded data, obtain from `prediction_data_preprocessing` method.
                
                keep_features: bool: if True, it will keep the features in the dataframe that will be saved as output CSV file.
                                     if False, it will drop the features from the dataframe that will be saved as output CSV file.
                                     Default: False
        
        Output:
            None, a CSV file will be saved in local directory with prediction label.

        Version: 0.0.1
        """
        try:
            # laoding original data
            prediction_output_path = os.path.join("./Prediction_data_from_DB", "prediction_data.csv")
            original_data = self.utils.read_csv_file(prediction_output_path) # return dataframe # kkep the original data
            
            # performing clustering on the data
            self.logger.info("Started prediction on batch data...")
            self.logger.info("Loading Clustering model")
            cluster = joblib.load("Models\kmeans_clustering\kmeans_clustering.pkl") # return model for clustering.
            self.logger.info("Clustering model loaded")

            # getting the cluster labels
            self.logger.info("Getting the cluster labels")
            cluster_labels = cluster.predict(data) # return array of cluster labels
            # adding cluster labels to the dataframe
            data["cluster_labels"] = cluster_labels
            self.logger.info("Cluster labels added to the dataframe")

            # getting pickle file from the directory as store it on list of models
            # # Creating empty list of models
            models_list = list() # 
            # looping thorugh all the files in the directory
            for root, folder, file in os.walk("./Models"): # looping through all the files in the directory.
                for filename in file:
                    if filename.endswith(".pkl"):
                        models_list.append(os.path.join(root, filename)) # path to full model file


            # creating empty list of storing all target labels
            target_prediction_list = list() # here all the targte will be stored as looping through all the cluster labels

            self.logger.info("Looping through each cluster Started...")
            unique_cluster_labels = data["cluster_labels"].unique() # return unique cluster labels
            # looping through each cluster
            for cluster_label in unique_cluster_labels:
                # getting the dataframe for the cluster
                cluster_df = data[data["cluster_labels"] == cluster_label] # return dataframe
                storing_index_number = list(cluster_df.index.values) # return list of index number
                
                # loading the model for the cluster from the list of models
                self.logger.info("Loading model for cluster: " + str(cluster_label))
                for model in models_list:
                    cluster_no = model.split(".pkl")[0].split("_")[1] # return cluster number only
                    if cluster_no == str(cluster_label): # if cluster number matches
                        model = joblib.load(model) # return model
                        self.logger.info(f"Model for cluster: {cluster_label} loaded")
                    else:
                        continue
                    # getting the prediction for the cluster
                    self.logger.info("Getting the prediction for the cluster: " + str(cluster_label))
                    # droping cluster labels from the dataframe
                    cluster_df = cluster_df.drop(["cluster_labels"], axis=1) # return dataframe
                    target_pred = list(model.predict(cluster_df)) # return list of predictions

                    # zipping it with index no before adding to the list
                    target_prediction_list.append(list(zip(storing_index_number, target_pred))) # return list of predictions
                    self.logger.info(f"Prediction for cluster: {cluster_label} obtained")

            # if user chhose to keep features
            if keep_features: # if user choose keep features.
                self.logger.info("User choose Keeping features in the dataframe")
                # adding extra column to raw prediction dataframe
                # unwraing targte prediction list, it's list of list with cluster count. 
                target_prediction_list = list(itertools.chain(*target_prediction_list)) # unwraping list.
                # now ablove list is list of tuples.                                                                                        
                
                # running loop to enter prediction in correct index value
                for index, prediction in target_prediction_list: # looping throug tupple:- (index, prediction_value)
                    original_data.loc[index, "Prediction"] = prediction # putting prediction into exact index position
                
                # saving the dataframe as csv file
                output_path = os.path.join("Prediction_Output_File", "Output_withFeature.csv")
                original_data.to_csv(output_path, 
                                    index=False)
                self.logger.info("Prediction output file saved with keeping features")

                self.logger.info(f"Prediction for cluster: {cluster_label} saved as CSV file")
            
            else: # if user chhose to drop features and keep prediction only.
                self.logger.info("User choose Dropping features in the dataframe")
                target_prediction_series = pd.Series(itertools.chain(*target_prediction_list)) # creating series from list
                # creaing empty dataframe with two columns, index and prediction
                prediction_df = pd.DataFrame(columns=["index", "Prediction"]) # return dataframe
                for index, prediction in target_prediction_series: # looping through series
                    prediction_df.loc[index, "Prediction"] = prediction # putting prediction into exact index position
                
                # saving the dataframe as csv file
                saved_output_path = os.path.join("Prediction_Output_File", "Output_OnlyPrediction.csv")
                prediction_df.to_csv(saved_output_path,
                                        index=True)
                self.logger.info("Prediction output file saved without keeping features")

            self.logger.info("Looping through each cluster Completed")

        except Exception as e:
            self.logger.error("Error occured while performing prediction on batch data" + str(e))
            raise Exception("Error occured while performing prediction on batch data" + str(e))

    
    
                




        
    

