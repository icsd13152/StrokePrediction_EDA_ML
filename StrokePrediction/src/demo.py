import joblib
import pandas as pd
import numpy as np

hd_model_obj = joblib.load('./models/WithoutChi2/GB_BMI_NoChi2.sav')
scaler = joblib.load('./models/WithoutChi2/scaler_BMI_NoChi2.sav')
# dataOriginal = pd.read_csv('../Dataset/1healthcare-dataset-stroke-data.csv')
# dataOriginal.head(10)
# # data = pd.read_csv('C:\Users\ppetropo\Desktop\AppsOFAI\StrokePrediction_EDA_ML\StrokePrediction\Dataset\healthcare-dataset-stroke-data.csv')
# # print(data.head())
column_names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level', 'bmi','smoking_status']
#
#
#
values = [0, 78, 0, 0, 1, 0, 0, 78.81, 19.6, 3]

x_patient = pd.DataFrame(data=[values],
                         columns=column_names,
                         index=[0])
x = scaler.transform(x_patient)
print(x_patient)

yhat = hd_model_obj.predict(x)
y_val = hd_model_obj.predict_proba(x)
print(y_val)
print(yhat)