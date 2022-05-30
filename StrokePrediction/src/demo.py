import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

hd_model_obj = joblib.load('./models/WithoutChi2/GB_BMI_NoChi2.sav')
scaler = joblib.load('./models/WithoutChi2/scaler_BMI_NoChi2.sav')
X_test = np.load('testsetX_as_nparray.npy')
y_test = np.load('testsetY_as_nparray.npy')
X_test = scaler.transform(X_test)
# dataOriginal = pd.read_csv('../Dataset/1healthcare-dataset-stroke-data.csv')
# dataOriginal.head(10)
# # data = pd.read_csv('C:\Users\ppetropo\Desktop\AppsOFAI\StrokePrediction_EDA_ML\StrokePrediction\Dataset\healthcare-dataset-stroke-data.csv')
# # print(data.head())
# column_names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level', 'bmi','smoking_status']
# #
# #
# #
# values = [0, 78, 0, 0, 1, 0, 0, 78.81, 19.6, 3]
#
# x_patient = pd.DataFrame(data=[values],
#                          columns=column_names,
#                          index=[0])
# x = scaler.transform(x_patient)
# print(x_patient)
#
# yhat = hd_model_obj.predict(x)
# y_val = hd_model_obj.predict_proba(x)
# print(y_val)
# print(yhat)

# clf_mean = GradientBoostingClassifier(n_estimators=300, learning_rate=1, loss = 'exponential', max_depth=10).fit(X_train, y_train)
# y_pred = clf_mean.predict(X_val)
# predicted_prob = clf_mean.predict_proba(X_val)
# m_confusion_test = metrics.confusion_matrix(y_val, y_pred)
# df_cm = pd.DataFrame(m_confusion_test, index = ['Stroke', 'Normal'],
#                      columns = ['Stroke', 'Normal'])
# print("GBC")
# print(metrics.classification_report(y_pred,y_val))
# sns.heatmap(df_cm,
#             annot=True,
#             cmap='Blues',
#             fmt='.5g',
#             annot_kws={"size": 20}).set_title('Confusion matrix', fontsize = 35, y=1.05)
# plt.xlabel('Predicted values', fontsize = 20)
# plt.ylabel('True values', fontsize = 20)
# plt.show()
clf_mean = hd_model_obj

for clf, label in zip([clf_mean],
                      [
                       'GradientBoosting'

                       ]):

    scoresTest = cross_val_score(clf, X_test, y_test,
                                 cv=5, scoring='accuracy')
    scores2Test = cross_val_score(clf, X_test, y_test,
                                  cv=5, scoring='f1')
    print("Accuracy on Test set: %0.3f (+/- %0.3f) [%s]"
          % (scoresTest.mean(), scoresTest.std(), label))
    print("F1-Score on Test set: %0.3f (+/- %0.3f) [%s]"
          % (scores2Test.mean(), scores2Test.std(), label))