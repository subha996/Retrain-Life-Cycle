# from data_validation.raw_data_validator import RawDataValidator


# raw = RawDataValidator(batch_directory="./Training_Batch_Files",
#                        validated_raw_directory="./Training_Validated_Raw_Files") 



# reg = "['insurance']+['\_'']+[\d_]+[\d]+\.csv"
# raw.validate_file_name(reg,
#                        8,
#                        6)
# raw.validateColumnLength(7)
# # raw.validateMissingValueWholeColumn()

# from training import Train


# tr = Train()

# tr.train()

import pandas as pd

d = {
    "age":19,
     "sex": "female",
     "bmi": 27.9,
     "children":2,
     "smoker": "yes",
     "region": "southeast",
}

df = pd.DataFrame(d, index=[0])

import joblib
trans = joblib.load("ColumnTransformer\column_transformer.pkl")

scal_data = trans.transform(df)
print(scal_data)

print(scal_data.shape)

clus = joblib.load("Models\kmeans_clustering\kmeans_clustering.pkl")
c_n = clus.predict(scal_data)
print(c_n[0])

model = joblib.load("Models\AdaBoostRegressor2\AdaBoostRegressor_2.pkl")

pred = model.predict(scal_data)
print(pred)

# SUCCESSS........................