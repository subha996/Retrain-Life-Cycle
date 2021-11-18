# Writing class for tuning various models to train data and fidn best from it.
# Date: 15-Nov-2021
# osprey was here...
#-------------------------------------#

# IMporting relevant libraries.
from Model_Finder.modelFinder import RegressionModelFinder
from applicationlogger.setup_logger import setup_logger
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from utils.fileoperation import DataGetter
import os


class ModelTuner():
    """
    This class shall use to find best model on train data.
    """
    def __init__(self, X_train, X_test, y_train, y_test, cluster_no):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cluster_no = cluster_no
        self.utils = DataGetter()
        self.model_finder = RegressionModelFinder(self.cluster_no) # Initializing model finder.
        self.logger = setup_logger("modelTunerLig",
                                    "model_tuner.log") # Initializing logger.
        self.logger.info("Model tuner initialized. " + "-"*50) # Season divider.
        

    # Writing function to check all model available in model finder. with train data and will return best model
    # Best will be saved on local directory.

    def run_model_tuner(self, cluster_number, on_score=True) -> None:
        """
        Method: run_model_tuner
        Descripton: This method shall use to check multiple models for regression problem.
                    The best finded model will store oon locala directory based on score that have chosen.
        
        Paramater:
                on_score: bool (default: True) - will select best model on score(r2_score)
                If False will choose best model on error(mean_squared_error)
        Output:
            None

        On Error: Log Error, rasie error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Model Tuner Started...")

            # Runnig all model from model finder class to train data and find best model.
            self.logger.info("Running all model from model finder class to train data and find best model...")
            
            # Running Linear Regression
            self.logger.info("Running Linear Regression...")
            lin_reg = self.model_finder.best_from_LinearRegression(self.X_train, self.y_train) # tupple: ("model_name", model_object)
            self.logger.info("Linear Regression Done.")

            # Running Ridge regression
            self.logger.info("Running Ridge regression...")
            ridge_reg = self.model_finder.best_from_Ridge(self.X_train, self.y_train) # return tuple of best model and score.
            self.logger.info("Ridge regression Done.")

            # Running Lasso regression
            self.logger.info("Running Lasso regression...")
            lasso_reg = self.model_finder.best_from_Lasso(self.X_train, self.y_train)
            self.logger.info("Lasso regression Done.")

            # Running Elastic Net regression
            self.logger.info("Running Elastic Net regression...")
            elastic_net_reg = self.model_finder.best_from_ElasticNet(self.X_train, self.y_train)
            self.logger.info("Elastic Net regression Done.")

            # Running Decession Tree
            self.logger.info("Running Decision Tree...")
            dec_tree = self.model_finder.best_from_DecisionTree(self.X_train, self.y_train)
            self.logger.info("Decision Tree Done.")

            # Running Random Forest
            self.logger.info("Running Random Forest...")
            random_forest = self.model_finder.best_from_RandomForest(self.X_train, self.y_train)
            self.logger.info("Random Forest Done.")
            
            # Running Gradient Boosting 
            self.logger.info("Running Gradient Boosting...")
            gradient_boosting = self.model_finder.best_from_GradientBoosting(self.X_train, self.y_train)
            self.logger.info("Gradient Boosting Done.")

            # Running AdaBoost
            self.logger.info("Running AdaBoost...")
            ada_boost = self.model_finder.best_from_AdaBoostRegressor(self.X_train, self.y_train)
            self.logger.info("AdaBoost Done.")

            # Running EctraTree Regressor.
            self.logger.info("Running ExtraTree Regressor...")
            extra_tree = self.model_finder.best_from_ExtraTreeRegressor(self.X_train, self.y_train)
            self.logger.info("ExtraTree Regressor Done.")

            # Running XGBoost
            self.logger.info("Running XGBoost...")
            xg_boost = self.model_finder.best_from_XGBoost(self.X_train, self.y_train)
            self.logger.info("XGBoost Done.")

            # Running SVR
            self.logger.info("Running SVR...")
            svr = self.model_finder.best_from_SVR(self.X_train, self.y_train)
            self.logger.info("SVR Done.")

            # Running Kenighbours Regressor
            self.logger.info("Running Kenighbours Regressor...")
            knn = self.model_finder.best_from_KNeighborsRegressor(self.X_train, self.y_train)
            self.logger.info("Kenighbours Regressor Done.")

            # Creating lsit of all tuple of best model and score.
            model_and_score = [lin_reg, 
                               ridge_reg, 
                               lasso_reg, 
                               elastic_net_reg, 
                               dec_tree, 
                               random_forest, 
                               gradient_boosting, 
                               ada_boost, 
                               extra_tree, 
                               xg_boost, 
                               svr, 
                               knn] # list of tupple of best model and score: 
                                    # ("model_name", model_object)
            
            # Looping throug all model to create test score. and loss adn store them in separate list.
            # Creating two empty list to store test score and loss.
            test_score = list() # empty list to store test score.
            test_loss = list() # empty list to store test loss.
            result_df = pd.DataFrame(index=["Score", "Loss"]) # empty dataframe to store test score and loss.

            self.logger.info("Calculating test score and loss for all model...")
            # looping through all model list
            for m_s in model_and_score:
                # Getting model name and model object from tuple.
                model_name = m_s[0] # first element from tuple
                model_object = m_s[1] # second element from tuple

                # Getting test score and loss from model object.
                test_pred = model_object.predict(self.X_test)
                score = r2_score(self.y_test, test_pred) # calculating test score.
                loss = mean_squared_error(self.y_test, test_pred) # calculating test loss.
                test_score.append(score) # appending test score to test score list.
                test_loss.append(loss) # appending test loss to test loss list.
                # adding test score and loss to dataframe.
                result_df[model_name] = [score, loss] # new column with model name for very iteration.
            self.logger.info("Calculating test score and loss for all model Done.")

            
            # saving result dataframe to disk.
            self.logger.info("Saving result dataframe to disk...")
            if not os.path.isdir("./Models_Results"):
                os.makedirs("./Models_Results")
            # os.makedirs("Models_Results", exist_ok=True) # creating directory if it not exist.
            path = os.path.join("Models_Results", "compare_models_for_cluster" + str(cluster_number) + ".csv")
            result_df.to_csv(path, index=False) # saving dataframe to disk.
            print("Saving result dataframe to disk Done.")
            self.logger.info("Saving result dataframe to disk Done.")
            
            # Getting best model on score.
            if on_score: # if user want to get best model on score.
                # getting the max score index number from test_score list.
                max_score_index = test_score.index(max(test_score)) # getting max score index number.
                model_name_to_save = model_and_score[max_score_index][0] # getting model name from model_and_score list.
                model_to_save = model_and_score[max_score_index][1] # getting best model object from model_and_score list.
                
                return model_name_to_save, model_to_save # returnig best model name and best model object.
            
            else: # if user want to get best model on loss.
                # getting the max loss index number from test_loss list.
                min_loss_index = test_loss.index(min(test_loss)) # getting min loss index number.
                model_name_to_save = model_and_score[min_loss_index][0] # getting model name from model_and_score list.
                model_to_save = model_and_score[min_loss_index][1] # getting best model object from model_and_score list.
                
                return model_name_to_save, model_to_save # returning best model name and best model object.

        except Exception as e:
            self.logger.error("Error in ModelFinder.best_model_finder() method. Error: {}".format(str(e)))
            raise Exception("Error in ModelFinder.best_model_finder() method. Error: {}".format(str(e)))


            
            











        