# Writing class for training model.
# Date: 16-Nov-2021
#--------------------------------------#

# imporing library
import os
from applicationlogger.setup_logger import setup_logger
from utils.fileoperation import DataGetter
from TrainingDataPreprocessing.trainingDataPreprocessing import TrainingDataPreprocessing
from TrainingDataPreprocessing.clustering import KMenasClustering
from sklearn.model_selection import train_test_split
from Model_Finder.modelTuner import ModelTuner



class Train():
    """
     THis class shall use for data pull from database, preprocess it
     and Train with multiple models to choose best from it.
     
    """
    def __init__(self):
        self.logger = setup_logger('Train', 'logs/train.log')
        self.utils = DataGetter() # object of DataGetter class
        self.trainPreprocessor = TrainingDataPreprocessing() # object of TrainingDataPreprocessing class
        self.clustering = KMenasClustering() # object of KMenasClustering class

    
    def preprocess(self):
        """
        This method shall be used to preprocess data.
        Paramaters:
            None
        Output:
            None
        
        On Error: Log Error.

        Version: 0.0.1
        """
        self.logger.info("Preprocess data...")
        self.trainPreprocessor.preprocess_training_data() # this will save preprocess data into local directory
        self.logger.info("Preprocess data completed.")

    def train(self):
        """
        This method shall be used to train data.
        Paramaters:
            None
        Output:
            None
        
        On Error: Log Error.

        Version: 0.0.1
        """
        try:
            self.logger.info("Train data has Started...")
            
            # Getting the preprocessed data
            data = self.trainPreprocessor.data_scaling_encoding()
            self.logger.info("Data Scaling and Encoding has completed...")

            # separating data into X and y
            X = data.drop(['expenses'], axis=1) # Drop the target column
            y = data['expenses']

            # getting the cluster number.
            self.logger.info("Getting the cluster number...")
            cluster_number = self.clustering.get_cluster_number(X) # return int
            self.logger.info("Getting the cluster number completed... Cluster Found:- " + str(cluster_number))

            # adding cluster to the feature dataframe
            data_with_cluster = self.clustering.create_cluster(X, cluster_number)
            

            # Adding target column to the dataframe
            data_with_cluster['expenses'] = y
            self.logger.info("Adding cluster to the feature dataframe completed.")

            # now looping through the cluster number and train the model
            self.logger.info(f"Total {cluster_number} has been found and start looping through all cluster...")
            for cluster in range(0, cluster_number):
                # getting the dataframe for cluster i
                cluster_data = data_with_cluster[data_with_cluster['cluster_label'] == cluster] # return DataFrame
                
                # getting the X and y data for cluster i
                X = cluster_data.drop(['expenses', 'cluster_label'], axis=1) # separating only features
                y = cluster_data['expenses'] # target column
                
                # splitting the data into train and test
                X_train, X_test, y_train, y_test = train_test_split(X, 
                                                                    y,
                                                                    test_size= 0.2)
                # training the model 
                # Creating object of model tuner classs
                tuner = ModelTuner(X_train, 
                                X_test, 
                                y_train, 
                                y_test)
                
                # getting the best model
                best_model_name, best_model = tuner.run_model_tuner(cluster_number=cluster,
                                                                    on_score=True # best model will be selected base on r2 score.
                                                                    )
                # saving the best model
                os.makedirs("./Models/" + best_model_name + str(cluster), exist_ok=True) # creating directory if not exist
                path = os.path.join('./Models', best_model_name + str(cluster), best_model_name)
                self.utils.write_pickle_file(best_model,
                                            path,
                                            cluster=cluster)
                self.logger.info("Model " + best_model_name + " for cluster " + str(cluster) + " has been saved.")
            self.logger.info("Train data has completed for all clusters.")

        except Exception as e:
            self.logger.error(f"Error occured while training: {str(e)}")
            raise Exception(e)
            
        

