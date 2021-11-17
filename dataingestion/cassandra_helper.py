# Creating class for database operation.
# Date: 30-Oct-2021
#----------------------------------------#

# importing necessary libraries
import os
import numpy as np
import pandas as pd
import csv
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import warnings
warnings.filterwarnings('ignore') # ignore warnings
from applicationlogger.setup_logger import setup_logger

# Database connection class
class CassandraHelper():
    """
    This class shall use to perform cassandra database operations.
    Methods:
        send_csv():
            This method shall be used to send csv file to cassandra database.
        get_data():
            This method shall be used to get data from cassandra database.
    """
    def __init__(self):
        
        # Setting up logger
        # Checking logs folder is exist or not
        if not os.path.isdir("./logs"):
            os.mkdir("./logs") # create logs folder if not exist
        self.log = setup_logger('cassandra_helper', "logs/cassandra_helper.log")

        try:
            self.log.info("Connecting to cassandra database.")
            self.cloud_config = {'secure_connect_bundle': 'secure-connect-database1.zip'} 
            self.auth_provider = PlainTextAuthProvider('sWSsgfrfxpukXnmXnHkxZbRI', 'YUcLBrvHMnq_piJZbhad,UQzPPX3NLKyWOg1pi9ar6KR78XIps774i6xK1DxDXmn9AJeBf-359uxfB.6w,fqmKajtG_8Ak,ZZ-nMOfxT7-cnHm,ghpOZw2GP82uGbAfc')
            self.cluster = Cluster(cloud=self.cloud_config, auth_provider=self.auth_provider) # Connecting to cassandra cluster
            # Keyspace should be created before using it.
            # self.keyspace = "insurance1"
            # self.session = self.cluster.connect(self.keyspace) # Connecting to database
            self.log.info("Connection established successfully.")
        except Exception as e:
            self.log.error("Error in connecting to database." + str(e))
            raise e

        
    def send_csv(self, file, keyspace, table_name):
        """
        This methode will upload CSV file from local to cloud storage.
        Parameter:
                 file: CSV file name
                 table_name: Table name in database. It will be create on cloud.

        Return:
                 None : Send data to the cloud from local.
        
        Version: 0.0.1
        """
        # Connecting to cluster
        self.log.info("Connecting to cluster")
        session = self.cluster.connect(keyspace) # COnnecting to database
        self.log.info("Connection is ready to Cluster")
        try:
            self.log.info("Entered send_csv methode...")
            # query for creating table
            create_table_query = f"create table {table_name}(column_count int, "
            # query for inserting data
            insert_into_table_query = f"insert into {table_name}(column_count , "
            # Creating list of column names with adding datatypes after "__dt__"
            columns = [col + "__dt__" + str(pd.read_csv(file)[col].dtype) for col in pd.read_csv(file).columns]

            # adding column names to query
            for col in columns:
                create_table_query += f"{col} text, "
                insert_into_table_query += f'{col}, ' 

            # Creatig table.
            try:
                # Checking table is already present or not
                call_table_query = f"select * from {keyspace}.{table_name};"
                try:
                    # Fetching all data from table
                    session.execute(call_table_query) # executing query
                    print("Table is present New table will be not created.")
                    self.log.info("Table is already present. Avoiding Creating New Table")
                except Exception as e:
                    self.log.info(f"Table is not present. Creating {table_name}")
                    self.log.info(f"Creating table on {keyspace} with {table_name}")
                    query = create_table_query.strip() + f" primary key(column_count));"
                    session.execute(query=query) # executing query to creating table   
                    self.log.info(f"Table {table_name} created successfully.")
            except Exception as e:
                self.log.error(f"Error in creating table {table_name}." + str(e))
                

            # Inserting data into table.
            try:
                self.log.info(f"Inserting data into {table_name}")
                query = insert_into_table_query.strip(", '") + ")" + " values(" + str((len(columns) + 1) * ('?,')).strip(", ") + ");"
                prepared_query = session.prepare(query) # preparing query
                
                # Loading primary key
                with open("./dataingestion/primary_key.txt", "r") as f:
                    primary_key = f.read()
                    f.close() # closing file
            
                primary_key = int(primary_key)# counter for column_count

                # opening the csv file
                with open(file, "r", errors='ignore') as f:
                    next(f) # skipping header
                    for line in f:
                        data = (line.strip("\n").replace('"', '').replace("'", "").split(","))
                        primary_key += 1 # incrementing column_count
                        # Writing primary key to file
                        with open("./dataingestion/primary_key.txt", "w") as f:
                            f.write(str(primary_key))
                            f.close() # closing file
                        data.insert(0, primary_key) # adding column_count
                        try:
                            session.execute(prepared_query, data) # executing query
                            print(f"Data inserted successfully into {table_name}")
                        except Exception as e:
                            self.log.error(f"Error in inserting data on {primary_key} row." + str(e))
                            raise e
                self.log.info(f"{primary_key} rows inserted to {table_name} table")
                self.log.info(f"Data inserted successfully. on {table_name} table")

            except Exception as e:
                self.log.error(f"Error in inserting data into {table_name}." + str(e))
                raise e
        
        except Exception as e:
            self.log.error(f"Error in send_csv methode." + str(e))
            raise e
        
        finally:
            self.log.info("Exiting send_csv methode... with clossing the seasson")
            if session and not session.is_shutdown:
                session.shutdown() # closing session
                self.log.info("Session is closed.")

    # Function to pull data from cloud
    def get_data(self, keyspace, table_name):
        """
        This methode will pull data from cloud to local.
        Parameter: 
                    keyspace: Keyspace name
                    table_name: Table name in database.
        Output:
                pandas DataFrame: Dataframe of data.
        Version: 0.0.1
        """
        # Connecting to cluster
        try:
            self.log.info("Connecting to cluster")
            session = self.cluster.connect(keyspace)
            self.log.info("Connection is ready to Cluster")
            self.log.info("Pulling data from cloud...")
            data_list = list(session.execute(f'select * from {table_name}'))
            self.log.info("Data pulled successfully.")
            df = pd.DataFrame(data_list).drop(columns=['column_count'], axis=1) # creating dataframe
            # changing datatypes
            for col in df.columns:
                col_name = col.split("__dt__")[0] # getting column name, before __dt__
                col_data_type = col.split("__dt__")[1] # getting datatype, after __dt__
                df.rename(columns={col: col_name}, inplace=True) # renaming column
                try:
                    df[col_name] = df[col_name].astype(col_data_type) # changing datatype
                except Exception as e:
                    self.log.error(f"Error in changing datatype of {col_name}." + str(e))
                    raise e
            self.log.info("Dataframe created successfully. Returning the data")
            return df.replace(r'^\s*$', np.nan, regex=True) # replacing empty values with NaN
        
        except Exception as e:
            self.log.error(f"Error in get_data methode." + str(e))
            raise e
        
        # Closing session
        finally:
            self.log.info("Exiting get_data methode... with clossing the seasson")
            if not session.is_shutdown:
                session.shutdown()
                self.log.info("Session is closed.")
                
