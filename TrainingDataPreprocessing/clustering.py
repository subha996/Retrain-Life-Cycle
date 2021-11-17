#write class for performing clustering on the traning data.
# Date: 16-Nov-2021
# --------------------------------------#

# Imorting libraries
import os
from applicationlogger.setup_logger import setup_logger
from utils.fileoperation import DataGetter
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt # for saving the elbow plot

class KMenasClustering():
    """
    This class shall be ised to divide the data into claster before traning.

    Version: 0.0.1
    """
    def __init__(self):
        self.logger = setup_logger("clustering_logs", "logs/Clustering.log") # setting up logger
        self.utils = DataGetter() # creating object of DataGetter class


    # Function to get cluster number
    def get_cluster_number(self, data):
        """
        This method shall be used to get the cluster number.

        Parameters: data: DataFrame without target column
        Output: cluster_number: int

        Version: 0.0.1
        """
        try:
            self.logger.info("Getting cluster number")
            
            wcss = [] # variable to store the value of wcss
            for i in range(1, 11): # getting the cluster number from 1 to 10
                kmeans = KMeans(n_clusters=i, 
                                init='k-means++', # initializing the kmeans
                                max_iter=300,  # maximum iteration
                                n_init=10, # number of initialization
                                random_state=32)
                kmeans.fit(data) # fitting the data
                wcss.append(kmeans.inertia_) # getting the value of wcss
            
            # creating plot of elbow method
            plt.plot(range(1, 11), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            # plt.show() # not showing in the console
            if not os.path.isdir("plots"): # checking if the directory is present
                os.makedirs("plots") # creating the directory
            file_to_save = os.path.join("plots", "elbow_plot.png") # getting the file path
            plt.savefig(file_to_save) # saving the plot
            
            self.logger.info("The elbow plot Saved to the local direcotry")
            # finding the elbow point thriugh keenelocator
            kn = KneeLocator(range(1, 11), 
                             wcss, 
                             curve='convex', 
                             direction='decreasing')
            cluster_number = kn.knee # getting the knee point
            self.logger.info(f"The elbow point found through KneeLocator: {cluster_number}")
            
            return cluster_number # returning the cluster number
        
        except Exception as e:
            self.logger.error(f"Error occured while getting the cluster number: {str(e)}")
            raise Exception(e)


    # Function to perform clustering
    def create_cluster(self, data, cluster_number):
        """
        This Method shall be used to performing clustering with appropriate cluster number.

        Parameters: data: DataFrame,
                    cluster_number: int
        
        Outputtype: cluster_data: DataFrame: adding extra column as `cluster_label`

        Version: 0.0.1
        """
        try:
            self.logger.info("Performing clustering")
            # creating the kmeans object
            kmeans = KMeans(n_clusters=cluster_number, 
                            init='k-means++', # initializing the kmeans
                            max_iter=300,  # maximum iteration
                            n_init=10, # number of initialization
                            random_state=32)
            cluster_data = kmeans.fit_predict(data) # fitting the data

            # adding the cluster label column
            data['cluster_label'] = cluster_data
            self.logger.info("Clustering performed successfully")

            # saving clustering alogrithms for later use(prediction time)
            os.makedirs("./Models/kmeans_clustering", exist_ok=True) # creating the directory
            file_path = os.path.join("Models", "kmeans_clustering", "kmeans_clustering")
            self.utils.write_pickle_file(kmeans, 
                                        file_path=file_path)
            return data # returning the cluster data
        
        except Exception as e:
            self.logger.error(f"Error occured while performing clustering: {str(e)}")
            raise Exception(e)
        
            

       