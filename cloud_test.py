from training_data_ingestion.train_data_ingestion import TrainDataIngestion



val = TrainDataIngestion("Training_Batch_Files")

# val.validate_training_data()
# val.data_transformation()

val.insert_data_into_database()






