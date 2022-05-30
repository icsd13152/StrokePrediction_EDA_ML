import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes, svm,metrics
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.preprocessing import MinMaxScaler

dataOriginal = pd.read_csv('../Dataset/healthcare-dataset-stroke-data.csv')
dataOriginal.head(10)

def FillWithMean(dataOriginal):
    data = dataOriginal
    data['bmi'] = data['bmi'].fillna(np.round(data.bmi.mean(), 2))
    return data

data = FillWithMean(dataOriginal)
data.drop(['id'], axis=1, inplace=True)
data['gender']=data['gender'].apply(lambda x: 1 if x=='Male' else 0)
data['ever_married']=data['ever_married'].apply(lambda x: 1 if x=='Yes' else 0)
data['Residence_type']=data['Residence_type'].apply(lambda x: 1 if x=='Urban' else 0)


def func1(x):
    if x=='Private':
        return 0
    elif x=='Self-employed':
        return 1
    elif x=='Govt-job':
        return 2
    elif x=='children':
        return 3
    else:
        return 4

def func2(x):
    if x=='formerly smoked':
        return 0
    elif x=='never smoked':
        return 1
    elif x=='smokes':
        return 2
    else:
        return 3

data['work_type']=data['work_type'].apply(func1)

data['smoking_status']=data['smoking_status'].apply(func2)

Xorig = data.drop(['stroke'], axis=1)
yorig = data.stroke
sm = SMOTE(random_state=0)
X, y = sm.fit_resample(Xorig, yorig)
print("Total: ", X.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42,stratify=y_train)

# np.save('testsetX_as_nparray.npy',X_test)
# np.save('testsetY_as_nparray.npy',y_test)

# scaler = MinMaxScaler()#StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_val = scaler.transform(X_val)


'''START Grid Search for best parameters'''

# list_alpha = np.arange(1/100000, 10, 0.1)
# param_grid = {'alpha':list_alpha
#               }
# grid = GridSearchCV(naive_bayes.ComplementNB(), param_grid, refit = True, verbose = 0)
# #ComplementNB()
#
# grid.fit(X_train, y_train)
# # print best parameter after tuning
# print(grid.best_params_)
# grid_predictions = grid.predict(X_val)
# #
# # # print classification report
# print(metrics.classification_report(y_val, grid_predictions))
#
#
#
# param_grid = {'C':[0.001, 0.01, 0.1, 1.0,1.1,1.2,10,100,1000],
#               'max_iter': [200,500,1000],
#               'solver':['newton-cg','lbfgs']
#               }
# grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose = 0)
#
#
# grid.fit(X_train, y_train)
# # print best parameter after tuning
# print(grid.best_params_)
# grid_predictions = grid.predict(X_val)
#
# # print classification report
# print(metrics.classification_report(y_val, grid_predictions))
#
# param_grid = {'n_estimators':[10,50,100,200,300,500],
#               'learning_rate': [0.001,0.01,0.1,1,10],
#               'max_depth':[10,20,50,70,100]
#               }
# grid = GridSearchCV(GradientBoostingClassifier(loss = 'exponential'), param_grid, refit = True, verbose = 0)
#
#
# grid.fit(X_train, y_train)
# # print best parameter after tuning
# print(grid.best_params_)
# grid_predictions = grid.predict(X_val)
#
# # print classification report
# print(metrics.classification_report(y_val, grid_predictions))
#
# param_grid = {'n_estimators':[10,50,100,200,300,500]
#
#               }
# grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 0)
#
#
# grid.fit(X_train, y_train)
# # print best parameter after tuning
# print(grid.best_params_)
# grid_predictions = grid.predict(X_val)
#
# # print classification report
# print(metrics.classification_report(y_val, grid_predictions))
#
# param_grid = {'C':[0.01,0.1,1,1.2,1.5,10,100,1000,2000,3000],
#               'kernel':['rbf'],
#               'gamma':['scale','auto']
#
#               }
# grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 2)
#
#
# grid.fit(X_train, y_train)
# # print best parameter after tuning
# print(grid.best_params_) #{'C': 2000, 'gamma': 'scale', 'kernel': 'rbf'}
# grid_predictions = grid.predict(X_val)
#
# # print classification report
# print(metrics.classification_report(y_val, grid_predictions))

'''END Grid Search for best parameters'''

#=============== Training START =====================

clf2_mean = naive_bayes.ComplementNB(alpha=1e-05).fit(X_train,y_train)
y_pred2 = clf2_mean.predict(X_val)
predicted_prob2 = clf2_mean.predict_proba(X_val)
m_confusion_test = metrics.confusion_matrix(y_val, y_pred2)
df_cm = pd.DataFrame(m_confusion_test, index = ['Stroke', 'Normal'],
                     columns = ['Stroke', 'Normal'])
print("NB")
print(metrics.classification_report(y_pred2,y_val))
plt.figure(figsize = (12,8))
sns.heatmap(df_cm,
            annot=True,
            cmap='Blues',
            fmt='.5g',
            annot_kws={"size": 20}).set_title('Confusion matrix', fontsize = 35, y=1.05)
plt.xlabel('Predicted values', fontsize = 20)
plt.ylabel('True values', fontsize = 20)
plt.show()
#================================================================================
# for BMI mean and without chi2{'C': 3000, 'gamma': 'scale', 'kernel': 'rbf'}
clf1_mean = svm.SVC(kernel='rbf',C=1.5,gamma='auto',probability=True).fit(X_train,y_train)
y_pred = clf1_mean.predict(X_val)
predicted_prob1 = clf1_mean.predict_proba(X_val)
m_confusion_test = metrics.confusion_matrix(y_val, y_pred)
df_cm = pd.DataFrame(m_confusion_test, index = ['Stroke', 'Normal'],
                     columns = ['Stroke', 'Normal'])
print("SVM RBF")
print(metrics.classification_report(y_pred,y_val))
sns.heatmap(df_cm,
            annot=True,
            cmap='Blues',
            fmt='.5g',
            annot_kws={"size": 20}).set_title('Confusion matrix', fontsize = 35, y=1.05)
plt.xlabel('Predicted values', fontsize = 20)
plt.ylabel('True values', fontsize = 20)
plt.show()

# ================================================================================
# for BMI mean and without chi2 {'C': 1.0, 'max_iter': 200, 'solver': 'newton-cg'}
clf3_mean = LogisticRegression(solver='newton-cg',C=1,max_iter=200).fit(X_train,y_train)
y_pred3 = clf3_mean.predict(X_val)
predicted_prob3 = clf3_mean.predict_proba(X_val)
m_confusion_test = metrics.confusion_matrix(y_val, y_pred3)
df_cm = pd.DataFrame(m_confusion_test, index = ['Stroke', 'Normal'],
                     columns = ['Stroke', 'Normal'])
print("Logistic Regression")
print(metrics.classification_report(y_pred3,y_val))
sns.heatmap(df_cm,
            annot=True,
            cmap='Blues',
            fmt='.5g',
            annot_kws={"size": 20}).set_title('Confusion matrix', fontsize = 35, y=1.05)
plt.xlabel('Predicted values', fontsize = 20)
plt.ylabel('True values', fontsize = 20)
plt.show()
#================================================================================
#for BMI mean and without chi2 {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 500}
clf_mean = GradientBoostingClassifier(n_estimators=500, learning_rate=1, loss = 'exponential', max_depth=10).fit(X_train, y_train)
y_pred = clf_mean.predict(X_val)
predicted_prob = clf_mean.predict_proba(X_val)
m_confusion_test = metrics.confusion_matrix(y_val, y_pred)
df_cm = pd.DataFrame(m_confusion_test, index = ['Stroke', 'Normal'],
                     columns = ['Stroke', 'Normal'])
print("GBC")
print(metrics.classification_report(y_pred,y_val))
sns.heatmap(df_cm,
            annot=True,
            cmap='Blues',
            fmt='.5g',
            annot_kws={"size": 20}).set_title('Confusion matrix', fontsize = 35, y=1.05)
plt.xlabel('Predicted values', fontsize = 20)
plt.ylabel('True values', fontsize = 20)
plt.show()

#==============================================================================
# for BMI mean and without chi2 {'n_estimators': 200}
forest_mean = RandomForestClassifier(n_estimators = 300).fit(X_train,y_train)
y_pred4 = forest_mean.predict(X_val)
predicted_prob4 = forest_mean.predict_proba(X_val)
m_confusion_test = metrics.confusion_matrix(y_val, y_pred4)
df_cm = pd.DataFrame(m_confusion_test, index = ['Stroke', 'Normal'],
                     columns = ['Stroke', 'Normal'])
print("Random Forest")
print(metrics.classification_report(y_pred4,y_val))
sns.heatmap(df_cm,
            annot=True,
            cmap='Blues',
            fmt='.5g',
            annot_kws={"size": 20}).set_title('Confusion matrix', fontsize = 35, y=1.05)
plt.xlabel('Predicted values', fontsize = 20)
plt.ylabel('True values', fontsize = 20)
plt.show()


for clf, label in zip([clf1_mean, clf2_mean,clf3_mean,clf_mean,forest_mean],
                      ['SVM',
                       'Naive Bayes',
                       'Logistic Regression',
                       'GradientBoosting',
                       'Random Forest'
                       ]):

    scoresTest = cross_val_score(clf, X_val, y_val,
                                 cv=3, scoring='accuracy')
    scores2Test = cross_val_score(clf, X_val, y_val,
                                  cv=3, scoring='f1')
    print("Accuracy on Test set: %0.3f (+/- %0.3f) [%s]"
          % (scoresTest.mean(), scoresTest.std(), label))
    print("F1-Score on Test set: %0.3f (+/- %0.3f) [%s]"
          % (scores2Test.mean(), scores2Test.std(), label))
# ================= Training END =====================================
scalerName = './models/WithoutChi2/scaler_BMI_NoChi2_noscaler.sav'
filenameLR_BMI_NoChi2 = './models/WithoutChi2/LR_BMI_NoChi2_noscaler.sav'
filenameSVM_BMI_NoChi2 = './models/WithoutChi2/SVM_BMI_NoChi2_noscaler.sav'
filenameRF_BMI_NoChi2 = './models/WithoutChi2/RF_BMI_NoChi2_noscaler.sav'
filenameGB_BMI_NoChi2 = './models/WithoutChi2/GB_BMI_NoChi2_noscaler.sav'
filenameNB_BMI_NoChi2 = './models/WithoutChi2/NB_BMI_NoChi2_noscaler.sav'
joblib.dump(clf3_mean, filenameLR_BMI_NoChi2)
joblib.dump(clf1_mean, filenameSVM_BMI_NoChi2)
joblib.dump(forest_mean, filenameRF_BMI_NoChi2)
joblib.dump(clf_mean, filenameGB_BMI_NoChi2)
joblib.dump(clf2_mean, filenameNB_BMI_NoChi2)
# joblib.dump(scaler, scalerName)