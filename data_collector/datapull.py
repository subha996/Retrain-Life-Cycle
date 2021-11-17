# Write class to pull data from database and transform it and save it to a  csv file in local directoy.
# Date: 16-Nov-2021
#--------------------------------------#

# importing libraries
import os
from applicationlogger.setup_logger import setup_logger
from dataingestion.cassandra_helper import CassandraHelper
from data_transformation.dataTransformation import DataTransformation
from utils.fileoperation import DataGetter


class DataPull():
    """
    Class for collect csv file from database and saved it in csv file in local
    """
    def __init__(self):
        self.logger = setup_logger("pull_logs", "data_collector/data_pull.log") # setting up ,ogger
        self.database = CassandraHelper() # creating object for cassandra helper class
        self.data_transformation = DataTransformation() # creating object for data transformation class
        self.utils = DataGetter() # creating object for data getter class


    def pull_data(self, keyspace: str, table: str, output_file: str):
        """
        This method will be used to pull data from database and transform it and save it to a  csv file in local directoy.
        
        Parameter: None
        Return: None

        Version: 0.0.1
        """
        try:
            self.logger.info("Started pulling data from database Started...")
            # getting data from database
            data = self.database.get_data(keyspace=keyspace,
                                          table_name=table) # return dataframe.

            # transforming data
            data =  self.data_transformation.replace_NULL_with_np_nan(data) # replace NULL with np.nan

            # saving data to csv file
            self.utils.write_csv_file(data, output_file) # saving data to csv file
            self.logger.info("Data pulled from database and saved to csv file.")

        except Exception as e:
            self.logger.error("Error while pulling data from database. Error: {}".format(e))
            raise e
            

      


