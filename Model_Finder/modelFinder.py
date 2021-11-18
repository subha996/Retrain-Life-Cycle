# Writing class to find the best model for the given data from multiple models for regression problem.
# Date: 14-Nov-2021
# Osprey was here...
#-----------------------------------------------------------------------------------------------------#
import os
from applicationlogger.setup_logger import setup_logger
# Importing libraries for the model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import r2_score
# from return_data import data
from utils.fileoperation import DataGetter





class RegressionModelFinder:
    def __init__(self, cluster_no):
        """
        Initialize the class with the logger.
        """
        self.logger = setup_logger("model_finder_log", 
                                    "logs/model_finder.log") # Logger
        self.utils = DataGetter() # Creating an object of DataGetter class
        self.cluster_no = cluster_no # Cluster number

    
    # Function to find the best from Linear Regression
    def best_from_LinearRegression(self, X_train, y_train) -> tuple:
        """
        Method: best_from_LinearRegression
        Description: This method will run LinearRegression on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best Linear Regression from GridSearch CV with best parameter.

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best Linear Regression")
            self.linear_regression = LinearRegression()
            # fitting the data on traning data
            self.linear_regression.fit(X_train, y_train)
            self.logger.info("Returning Best Linear Regression.")
            # return linear regression
            return ("Linear Regression", self.linear_regression) # return linear regression trained model
        except Exception as e:
            print("Error in Running Linear Regression Model" + str(e))
            self.logger.error("Error in Finding best model from linear regression." + str(e))
            raise e
            

    def best_from_Ridge(self, X_train, y_train)-> tuple:
        """
        Method: best_from_Ridge
        Description: This method will run Ridge on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best Ridge from GridSearch CV with best parameter.
                Name of the Algorithm(Ridge)

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best for Ridge...")
            params = self.utils.read_yaml_file("params\Ridge_params.yaml") # Reading the parameters from yaml file: dictioanry
            
            self.ridge = Ridge() # creating an object of Ridge
            # Creating grid search object
            grid_search = GridSearchCV(self.ridge, # Ridge Regression
                                       params,  # parameters
                                       cv=2, # Cross Validation
                                       scoring="r2", # Scoring
                                       verbose=4 # Verbosity
                                       )
            
            # fitting the data on traning data
            grid_search.fit(X_train, y_train)

            # Extracting the best parameters and best estimator and save them in yaml file
            best_params = grid_search.best_params_ # Return dictionary of best parameters
            alpha = best_params["alpha"]
            fit_intercept = best_params["fit_intercept"]
            normalize = best_params["normalize"]
            solver = best_params["solver"]
            tol = best_params["tol"]
            max_iter = best_params["max_iter"]
            
            # Writing the best parameters in yaml file
            best_params_path = os.path.join("BestParams", "Ridge_Cluster_", str(self.cluster_no) + "Ridge.yaml")
            # making directory if not exists
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            self.utils.write_yaml_file(best_params_path, best_params)
            self.logger.info("Best Parameter value are stored in yaml file.")
            # Creating model with best parameters
            ridge_best = Ridge(alpha=alpha,
                                fit_intercept=fit_intercept,
                                normalize=normalize,
                                solver=solver,
                                tol=tol,
                                max_iter=max_iter
                                )
            
            # fitting the data on traning data
            ridge_best.fit(X_train, y_train) # fitting the data on traning data
            self.logger.info("Returning Best Ridge after train with traning data.")
            print("Ridge Regression Done" + "-"*50) # season divider. Removed in time of deployment.
            return ("Ridge", ridge_best) # return ridge trained model with best parameters found from grid search

        except Exception as e:
            print("Error in Running Ridge Model" + str(e))
            self.logger.error("Error in Running Ridge Model" + str(e))
            raise e

    # Function to find the best from Lasso
    def best_from_Lasso(self, X_train, y_train):
        """
        Method: best_from_Lasso
        Description: This method will run Lasso on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best Lasso from GridSearch CV with best parameter.
                Name of the Algorithm(Lasso)

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best for Lasso...")
            params = self.utils.read_yaml_file("params\Lasso_params.yaml") # Reading the parameters from yaml file: dictioanry

            self.lasso = Lasso() # creating an object of Lasso
            # Creating grid search object
            grid_search = GridSearchCV(self.lasso, # Lasso Regression
                                       params,  # parameters
                                       cv=2, # Cross Validation
                                       scoring="r2", # Scoring
                                       verbose=4 # Verbosity
                                       )
            
            # fitting the data on traning data
            grid_search.fit(X_train, y_train)

            # Extracting the best parameters and best estimator and save them in yaml file
            best_params = grid_search.best_params_ # Return dictionary of best parameters
            alpha = best_params["alpha"]
            fit_intercept = best_params["fit_intercept"]
            normalize = best_params["normalize"]
            positive = best_params["positive"]
            tol = best_params["tol"]
            max_iter = best_params["max_iter"]
            selection = best_params["selection"]
            
            # Writing the best parameters in yaml file
            best_params_path = os.path.join("BestParams", "Lasso_Cluster_", str(self.cluster_no) + "Lasso.yaml")
            # making directory if not exists
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            self.utils.write_yaml_file(best_params_path, best_params)
            self.logger.info("Best Parameter value are stored in yaml file.")

            # Creating model with best parameters
            lasso_best = Lasso(alpha=alpha,
                            fit_intercept=fit_intercept,
                            normalize=normalize,
                            selection=selection,
                            tol=tol,
                            max_iter=max_iter,
                            positive=positive
                            )
            
            # fitting the data on traning data
            lasso_best.fit(X_train, y_train) # fitting the data on traning data
            self.logger.info("Returning Best Lasso after train with train data.")

            print("Lasso Regression Done" + "-"*50) # season divider. Removed in time of deployment.
            return ("Lasso", lasso_best) # return lasso trained model with best parameters found from grid search

        except Exception as e:
            print("Error in Running Lasso Model" + str(e))
            self.logger.error("Error in Running Lasso Model" + str(e))
            raise e


    # Function to find the best from ElasticNet
    def best_from_ElasticNet(self, X_train, y_train) -> tuple:
        """
        Method: best_from_ElasticNet
        Description: This method will run ElasticNet on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best ElasticNet from GridSearch CV with best parameter.
                Name of the Algorithm(ElasticNet)

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best for ElasticNet...")
            params = self.utils.read_yaml_file("params\ElasticNet_params.yaml") # Reading the parameters from yaml file: dictioanry

            self.elastic_net = ElasticNet() # creating an object of ElasticNet
            # Creating grid search object
            grid_search = GridSearchCV(self.elastic_net, # ElasticNet Regression
                                       params,  # parameters
                                       cv=2, # Cross Validation
                                       scoring="r2", # Scoring
                                       verbose=4 # Verbosity
                                       )
            
            # fitting the data on traning data
            grid_search.fit(X_train, y_train)

            # Extracting the best parameters and best estimator and save them in yaml file
            best_params = grid_search.best_params_ # Return dictionary of best parameters
            alpha = best_params["alpha"]
            l1_ratio = best_params["l1_ratio"]
            fit_intercept = best_params["fit_intercept"]
            normalize = best_params["normalize"]
            tol = best_params["tol"]
            max_iter = best_params["max_iter"]
            positive = best_params["positive"]
            selection = best_params["selection"]
            
            # Writing the best parameters in yaml file
            best_params_path = os.path.join("BestParams", "ElasticNet_Cluster_", str(self.cluster_no) + "ElasticNet.yaml")
            # making directory if not exists
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            self.utils.write_yaml_file(best_params_path, best_params)
            self.logger.info("Best Parameter value are stored in yaml file.")

            # Creating model with best parameters
            elastic_net_best = ElasticNet(alpha=alpha,
                            l1_ratio=l1_ratio,
                            fit_intercept=fit_intercept,
                            normalize=normalize,
                            selection=selection,
                            tol=tol,
                            max_iter=max_iter,
                            positive=positive
                            )
            
            # fitting the data on traning data
            elastic_net_best.fit(X_train, y_train) # fitting the data on traning data
            self.logger.info("Returning Best ElasticNet after train with train data.")

            print("ElasticNet Regression Done" + "-"*50) # season divider. Removed in time of deployment.
            return ("ElasticNet", elastic_net_best) # return elastic net trained model with best parameters found from grid search

        except Exception as e:
            print("Error in Running ElasticNet Model" + str(e))
            self.logger.error("Error in Running ElasticNet Model" + str(e))
            raise e

    
    # Function to find the best form DecisionTree
    def best_from_DecisionTree(self, X_train, y_train) -> tuple:
        """
        Method: best_from_DecisionTree
        Description: This method will run DecisionTree on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best DecisionTree from GridSearch CV with best parameter.
                Name of the Algorithm(DecisionTree)

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best for DecisionTree...")
            params = self.utils.read_yaml_file("params\DecisionTreeRegressor_params.yaml") # Reading the parameters from yaml file: dictioanry

            self.decision_tree = DecisionTreeRegressor() # creating an object of DecisionTreeRegressor
            # Creating grid search object
            grid_search = GridSearchCV(self.decision_tree, # DecisionTree Regression
                                       params,  # parameters
                                       cv=2, # Cross Validation
                                       scoring="r2", # Scoring
                                       verbose=4 # Verbosity
                                       )
            
            # fitting the data on traning data
            grid_search.fit(X_train, y_train)

            # Extracting the best parameters and best estimator and save them in yaml file
            best_params = grid_search.best_params_ # Return dictionary of best parameters
            criterion = best_params["criterion"]
            splitter = best_params["splitter"]
            max_depth = best_params["max_depth"]
            min_samples_split = best_params["min_samples_split"]
            min_samples_leaf = best_params["min_samples_leaf"]
            max_features = best_params["max_features"]

            # Writing the best parameters in yaml file
            best_params_path = os.path.join("BestParams", "DecisionTree_Cluster_", str(self.cluster_no) + "DecisionTree.yaml")
            # making directory if not exists
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            self.utils.write_yaml_file(best_params_path, best_params)
            self.logger.info("Best Parameter value are stored in yaml file.")

            # Creating model with best parameters
            decision_tree_best = DecisionTreeRegressor(criterion=criterion,
                            splitter=splitter,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features
                            )
            
            # fitting the data on traning data
            decision_tree_best.fit(X_train, y_train) # fitting the data on traning data
            self.logger.info("Returning Best DecisionTree after train with train data.")

            print("DecisionTree Regression Done" + "-"*50) # season divider. Removed in time of deployment.
            return ("DecisionTree", decision_tree_best) # return decision tree trained model with best parameters found from grid search

        except Exception as e:
            print("Error in Running DecisionTree Model" + str(e))
            self.logger.error("Error in Running DecisionTree Model" + str(e))
            raise e

        
    # Function to find the best from RandomForest
    def best_from_RandomForest(self, X_train, y_train) -> tuple:
        """
        Method: best_from_RandomForest
        Description: This method will run RandomForest on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best RandomForest from GridSearch CV with best parameter.
                Name of the Algorithm(RandomForest)

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best for RandomForest...")
            params = self.utils.read_yaml_file("params\RandomForestRegressor_params.yaml") # Reading the parameters from yaml file: dictioanry

            self.random_forest = RandomForestRegressor() # creating an object of RandomForestRegressor
            # Creating grid search object
            grid_search = GridSearchCV(self.random_forest, # RandomForest Regression
                                       params,  # parameters
                                       cv=2, # Cross Validation
                                       scoring="r2", # Scoring
                                       verbose=4 # Verbosity
                                       )
            
            # fitting the data on traning data
            grid_search.fit(X_train, y_train)

            # Extracting the best parameters and best estimator and save them in yaml file
            best_params = grid_search.best_params_ # Return dictionary of best parameters
            n_estimators = best_params["n_estimators"]
            criterion = best_params["criterion"]
            max_depth = best_params["max_depth"]
            min_samples_split = best_params["min_samples_split"]
            min_samples_leaf = best_params["min_samples_leaf"]
            max_features = best_params["max_features"]
            bootstrap =  best_params["bootstrap"]
            oob_score = best_params["oob_score"]

            # Writing the best parameters in yaml file
            best_params_path = os.path.join("BestParams", "RandomForest_Cluster_", str(self.cluster_no) + "RandomForest.yaml")
            # making directory if not exists
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            self.utils.write_yaml_file(best_params_path, best_params)
            self.logger.info("Best Parameter value are stored in yaml file.")

            # Creating model with best parameters
            random_forest_best = RandomForestRegressor(n_estimators=n_estimators,
                            criterion=criterion,    
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            bootstrap=bootstrap,
                            oob_score=oob_score
                            )

            # fitting the data on traning data
            random_forest_best.fit(X_train, y_train) # fitting the data on traning data
            self.logger.info("Returning Best RandomForest after train with train data.")

            print("RandomForest Regression Done" + "-"*50) # season divider. Removed in time of deployment.
            return ("RandomForest", random_forest_best) # return random forest trained model with best parameters found from grid search

        except Exception as e:
            print("Error in Running RandomForest Model" + str(e))
            raise e

    # Function to find the best from GradientBoosting
    def best_from_GradientBoosting(self, X_train, y_train) -> tuple:
        """
        Method: best_from_GradientBoosting
        Description: This method will run GradientBoosting on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best GradientBoosting from GridSearch CV with best parameter.
                Name of the Algorithm(GradientBoosting)

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best for GradientBoosting...")
            params = self.utils.read_yaml_file("params\GradientBoostRegressor_params.yaml") # Reading the parameters from yaml file: dictioanry

            self.gradient_boosting = GradientBoostingRegressor() # creating an object of GradientBoostingRegressor
            # Creating grid search object
            grid_search = GridSearchCV(self.gradient_boosting, # GradientBoosting Regression
                                       params,  # parameters
                                       cv=2, # Cross Validation
                                       scoring="r2", # Scoring
                                       verbose=4 # Verbosity
                                       )
            
            # fitting the data on traning data
            grid_search.fit(X_train, y_train)

            # Extracting the best parameters and best estimator and save them in yaml file
            best_params = grid_search.best_params_ # Return dictionary of best parameters
            loss = best_params["loss"]
            learning_rate = best_params["learning_rate"]
            n_estimators = best_params["n_estimators"]
            criterion = best_params["criterion"]
            max_depth = best_params["max_depth"]
            min_samples_split = best_params["min_samples_split"]
            max_features = best_params["max_features"]

            # Writing the best parameters in yaml file
            best_params_path = os.path.join("BestParams", "GradientBoost_Cluster_", str(self.cluster_no) + "GradientBoost.yaml")
            # making directory if not exists
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            self.utils.write_yaml_file(best_params_path, best_params)

            self.logger.info("Best Parameter value are stored in yaml file.")

            # Creating model with best parameters
            gradient_boosting_best = GradientBoostingRegressor(loss=loss,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators,
                            criterion=criterion,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            max_features=max_features
                            )

            # fitting the data on traning data
            gradient_boosting_best.fit(X_train, y_train) # fitting the data on traning data
            self.logger.info("Returning Best GradientBoosting after train with train data.")

            print("GradientBoosting Regression Done" + "-"*50) # season divider. Removed in time of deployment.
            return ("GradientBoosting", gradient_boosting_best) # return gradient boosting trained model with best parameters found from grid search

        except Exception as e:
            print("Error in Running GradientBoosting Model" + str(e))
            self.logger.error("Error in Running GradientBoosting Model" + str(e))
            raise e

    
    # Function to find the best from AdaBoost Regressor
    def best_from_AdaBoostRegressor(self, X_train, y_train) -> tuple:
        """
        Method: best_from_AdaBoostRegressor
        Description: This method will run AdaBoostRegressor on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best AdaBoostRegressor from GridSearch CV with best parameter.
                Name of the Algorithm(AdaBoostRegressor)

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best for AdaBoostRegressor...")
            params = self.utils.read_yaml_file("params\AdaBoostRegressor_params.yaml") # Reading the parameters from yaml file: dictioanry

            self.ada_boost_regressor = AdaBoostRegressor() # creating an object of AdaBoostRegressor
            # Creating grid search object
            grid_search = GridSearchCV(self.ada_boost_regressor, # AdaBoostRegressor Regression
                                       params,  # parameters
                                       cv=2, # Cross Validation
                                       scoring="r2", # Scoring
                                       verbose=4 # Verbosity
                                       )
            
            # fitting the data on traning data
            grid_search.fit(X_train, y_train)

            # Extracting the best parameters and best estimator and save them in yaml file
            best_params = grid_search.best_params_ # Return dictionary of best parameters
            loss = best_params["loss"]
            learning_rate = best_params["learning_rate"]
            n_estimators = best_params["n_estimators"]

            # Writing the best parameters in yaml file
            best_params_path = os.path.join("BestParams", "AdaBoost_Cluster_", str(self.cluster_no) + "AdaBoost.yaml")
            # making directory if not exists
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            self.utils.write_yaml_file(best_params_path, best_params)

            self.logger.info("Best Parameter value are stored in yaml file.")

            # Creating model with best parameters
            ada_boost_regressor_best = AdaBoostRegressor(loss=loss,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators
                            )

            # fitting the data on traning data
            ada_boost_regressor_best.fit(X_train, y_train) # fitting the data on traning data
            self.logger.info("Returning Best AdaBoostRegressor after train with train data.")

            print("AdaBoostRegressor Regression Done" + "-"*50) # season divider. Removed in time of deployment.
            return ("AdaBoostRegressor", ada_boost_regressor_best) # return ada boost trained model with best parameters found from grid search

        except Exception as e:
            print("Error in Running AdaBoostRegressor Model" + str(e))
            self.logger.error("Error in Running AdaBoostRegressor Model" + str(e))
            raise e

    
    # Frunction to find the best from ExtraTreeRegressor
    def best_from_ExtraTreeRegressor(self, X_train, y_train) -> tuple:
        """
        Method: best_from_ExtraTreeRegressor
        Description: This method will run ExtraTreeRegressor on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best ExtraTreeRegressor from GridSearch CV with best parameter.
                Name of the Algorithm(ExtraTreeRegressor)

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best for ExtraTreeRegressor...")
            params = self.utils.read_yaml_file("params\ExtraTreeRegressor_params.yaml") # Reading the parameters from yaml file: dictioanry

            self.extra_tree_regressor = ExtraTreesRegressor() # creating an object of ExtraTreeRegressor
            # Creating grid search object
            grid_search = GridSearchCV(self.extra_tree_regressor, # ExtraTreeRegressor Regression
                                       params,  # parameters
                                       cv=2, # Cross Validation
                                       scoring="r2", # Scoring
                                       verbose=4 # Verbosity
                                       )
            
            # fitting the data on traning data
            grid_search.fit(X_train, y_train)

            # Extracting the best parameters and best estimator and save them in yaml file
            best_params = grid_search.best_params_ # Return dictionary of best parameters
            n_estimators = best_params["n_estimators"]
            criterion = best_params["criterion"]
            max_depth = best_params["max_depth"]
            min_samples_split = best_params["min_samples_split"]
            max_features = best_params["max_features"]
            bootstrap = best_params["bootstrap"]
            oob_score = best_params["oob_score"]

            # Writing the best parameters in yaml file
            best_params_path = os.path.join("BestParams", "ExtraTree_Cluster_", str(self.cluster_no) + "ExtraTree.yaml")
            # making directory if not exists
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            self.utils.write_yaml_file(best_params_path, best_params)
            self.logger.info("Best Parameter value are stored in yaml file.")

            # Creating model with best parameters
            extra_tree_regressor_best = ExtraTreesRegressor(n_estimators=n_estimators,
                            criterion=criterion,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            max_features=max_features,
                            bootstrap=bootstrap,
                            oob_score=oob_score
                            )
            
            # fitting the data on traning data
            extra_tree_regressor_best.fit(X_train, y_train) # fitting the data on traning data
            self.logger.info("Returning Best ExtraTreeRegressor after train with train data.")

            print("ExtraTreeRegressor Regression Done" + "-"*50) # season divider. Removed in time of deployment.
            return ("ExtraTreeRegressor", extra_tree_regressor_best) # return extra tree trained model with best parameters found from grid search

        except Exception as e:
            print("Error in Running ExtraTreeRegressor Model" + str(e))
            self.logger.error("Error in Running ExtraTreeRegressor Model" + str(e))
            raise e

    
    # Function to find the best from XGBoost
    def best_from_XGBoost(self, X_train, y_train) -> tuple:
        """
        Method: best_from_XGBoost
        Description: This method will run XGBoost on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best XGBoost from GridSearch CV with best parameter.
                Name of the Algorithm(XGBoost)

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best for XGBoost...")
            params = self.utils.read_yaml_file("params\XGBRegressor_params.yaml") # Reading the parameters from yaml file: dictioanry

            self.xgboost = xgb.XGBRegressor() # creating an object of XGBoostRegressor
            # Creating grid search object
            grid_search = GridSearchCV(self.xgboost, # XGBoost Regression
                                       params,  # parameters
                                       cv=2, # Cross Validation
                                       scoring="r2", # Scoring
                                       verbose=4 # Verbosity
                                       )
            
            # fitting the data on traning data
            grid_search.fit(X_train, y_train)

            # Extracting the best parameters and best estimator and save them in yaml file
            best_params = grid_search.best_params_ # Return dictionary of best parameters
            learning_rate = best_params["learning_rate"]
            n_estimators = best_params["n_estimators"]
            max_depth = best_params["max_depth"]

            # Writing the best parameters in yaml file
            best_params_path = os.path.join("BestParams", "XGBoost_Cluster_", str(self.cluster_no) + "XGBoost.yaml")
            # making directory if not exists
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            self.utils.write_yaml_file(best_params_path, best_params)
            self.logger.info("Best Parameter value are stored in yaml file.")

            # Creating model with best parameters
            xgboost_best = xgb.XGBRegressor(learning_rate=learning_rate,
                            n_estimators=n_estimators,
                            max_depth=max_depth
                            )

            # fitting the data on traning data
            xgboost_best.fit(X_train, y_train) # fitting the data on traning data
            self.logger.info("Returning Best XGBoost after train with train data.")

            print("XGBoost Regression Done" + "-"*50) # season divider. Removed in time of deployment.
            return ("XGBoost", xgboost_best) # return xgboost trained model with best parameters found from grid search

        except Exception as e:
            print("Error in Running XGBoost Model" + str(e))
            self.logger.error("Error in Running XGBoost Model" + str(e))
            raise e

    # Function to find the best from Support Vector Regressor
    def best_from_SVR(self, X_train, y_train) -> tuple:
        """
        Method: best_from_SVR
        Description: This method will run Support Vector Regressor on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best Support Vector Regressor from GridSearch CV with best parameter.
                Name of the Algorithm(Support Vector Regressor)

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best for Support Vector Regressor...")
            params = self.utils.read_yaml_file("params\SVR_params.yaml") # Reading the parameters from yaml file: dictioanry

            self.svr = SVR() # creating an object of Support Vector Regressor
            # Creating grid search object
            grid_search = GridSearchCV(self.svr, # Support Vector Regressor
                                       params,  # parameters
                                       cv=2, # Cross Validation
                                       scoring="r2", # Scoring
                                       verbose=4 # Verbosity
                                       )
            
            # fitting the data on traning data
            grid_search.fit(X_train, y_train)

            # Extracting the best parameters and best estimator and save them in yaml file
            best_params = grid_search.best_params_ # Return dictionary of best parameters
            kernel = best_params["kernel"]
            degree = best_params["degree"]
            gamma = best_params["gamma"]
            coef0 = best_params["coef0"]
            tol = best_params["tol"]
            max_iter = best_params["max_iter"]
            shrinking = best_params["shrinking"]
            C = best_params["C"]
           

            # Writing the best parameters in yaml file
            best_params_path = os.path.join("BestParams", "SVR_Cluster_", str(self.cluster_no) + "SVR.yaml")
            # making directory if not exists
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            self.utils.write_yaml_file(best_params_path, best_params)
            self.logger.info("Best Parameter value are stored in yaml file.")

            # Creating model with best parameters
            svr_best = SVR(C=C,
                            kernel=kernel,
                            degree=degree,
                            gamma=gamma,
                            coef0=coef0,
                            tol=tol,
                            max_iter=max_iter,
                            shrinking=shrinking
                            )
            
            # fitting the data on traning data
            svr_best.fit(X_train, y_train) # fitting the data on traning data
            self.logger.info("Returning Best Support Vector Regressor after train with train data.")

            print("Support Vector Regressor Regression Done" + "-"*50) # season divider. Removed in time of deployment.

            return ("Support Vector Regressor", svr_best) # return Support Vector Regressor trained model with best parameters found from grid search

        except Exception as e:
            print("Error in Running Support Vector Regressor Model" + str(e))
            self.logger.error("Error in Running Support Vector Regressor Model" + str(e))
            raise e

    # Function to find the best from KNeighborsRegressor
    def best_from_KNeighborsRegressor(self, X_train, y_train) -> tuple:
        """
        Method: best_from_KNeighborsRegressor
        Description: This method will run KNeighborsRegressor on traning data with GridSearchCV.
        
        Input: X_train, y_train
        Output: best KNeighborsRegressor from GridSearch CV with best parameter.
                Name of the Algorithm(KNeighborsRegressor)

        On Error: Log Error.

        Version: 0.0.1

        """
        try:
            self.logger.info("Finding best for KNeighborsRegressor...")
            params = self.utils.read_yaml_file("params\KNeighborsRegressor_params.yaml") # Reading the parameters from yaml file: dictioanry

            self.knr = KNeighborsRegressor() # creating an object of KNeighborsRegressor
            # Creating grid search object
            grid_search = GridSearchCV(self.knr, # KNeighborsRegressor
                                       params,  # parameters
                                       cv=2, # Cross Validation
                                       scoring="r2", # Scoring
                                       verbose=4 # Verbosity
                                       )
            
            # fitting the data on traning data
            grid_search.fit(X_train, y_train)

            # Extracting the best parameters and best estimator and save them in yaml file
            best_params = grid_search.best_params_ # Return dictionary of best parameters
            n_neighbors = best_params["n_neighbors"]
            algorithm = best_params["algorithm"]
            leaf_size = best_params["leaf_size"]
            p = best_params["p"]

            # Writing the best parameters in yaml file
            best_params_path = os.path.join("BestParams", "KNeighborsRegressor_Cluster_", str(self.cluster_no) + "KNeighborsRegressor.yaml")
            # making directory if not exists
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            self.logger.info("Best Parameter value are stored in yaml file.")

            # Creating model with best parameters
            knr_best = KNeighborsRegressor(n_neighbors=n_neighbors,
                            algorithm=algorithm,
                            leaf_size=leaf_size,
                            p=p
                            )
            
            # fitting the data on traning data
            knr_best.fit(X_train, y_train) # fitting the data on traning data
            self.logger.info("Returning Best KNeighborsRegressor after train with train data.")

            print("KNeighborsRegressor Regression Done" + "-"*50) # season divider. Removed in time of deployment.

            return ("KNeighborsRegressor", knr_best) # return KNeighborsRegressor trained model with best parameters found from grid search

        except Exception as e:
            print("Error in Running KNeighborsRegressor Model" + str(e))
            self.logger.error("Error in Running KNeighborsRegressor Model" + str(e))
            raise e









                                    

            
        
    



    