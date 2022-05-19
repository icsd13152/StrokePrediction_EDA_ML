import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn import naive_bayes, svm, metrics
from sklearn import preprocessing, feature_selection
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
# !pip install missingno
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

dataOriginal = pd.read_csv('../Dataset/healthcare-dataset-stroke-data.csv')
dataOriginal.head(10)

lr = LinearRegression()
datanew = dataOriginal[['age', 'ever_married', 'smoking_status','avg_glucose_level','bmi']].copy()

traindf = datanew[datanew['bmi'].isnull()==False]
testdf = datanew[datanew['bmi'].isnull()==True]
y = traindf['bmi']
traindf.drop("bmi",axis=1,inplace=True)
cat_columns = [cname for cname in traindf.columns
               if traindf[cname].dtype == 'object']

encoder = preprocessing.LabelEncoder()

for col in cat_columns:
    traindf[col] = encoder.fit_transform(traindf[col])

cat_columns = [cname for cname in testdf.columns
               if testdf[cname].dtype == 'object']

encoder = preprocessing.LabelEncoder()

for col in cat_columns:
    testdf[col] = encoder.fit_transform(testdf[col])
lr.fit(traindf,y)
testdf.drop("bmi",axis=1,inplace=True)
pred = lr.predict(testdf)
testdf['bmi']= pred

def FillWithLR(dataOriginal):
    data = dataOriginal
    # data[data['bmi'].isnull==True] = testdf['bmi']
    data['bmi'] = data['bmi'].fillna(testdf['bmi'])

    return data
data = FillWithLR(dataOriginal)

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

print("train: ",X_train.shape)
print("val: ",X_val.shape)
print("test: ",X_test.shape)

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

clf2_regre = naive_bayes.ComplementNB(alpha=1e-05).fit(X_train,y_train)
y_pred2 = clf2_regre.predict(X_val)
predicted_prob2 = clf2_regre.predict_proba(X_val)
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

clf1_regre = svm.SVC(kernel='rbf',C=1.5,gamma='auto',probability=True).fit(X_train,y_train)
y_pred = clf1_regre.predict(X_val)
predicted_prob1 = clf1_regre.predict_proba(X_val)
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

clf3_regre = LogisticRegression(solver='lbfgs',C=1,max_iter=200).fit(X_train,y_train)
y_pred3 = clf3_regre.predict(X_val)
predicted_prob3 = clf3_regre.predict_proba(X_val)
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
clf_regre = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, loss = 'exponential', max_depth=10).fit(X_train, y_train)
y_pred = clf_regre.predict(X_val)
predicted_prob = clf_regre.predict_proba(X_val)
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
#for BMI mean and without chi2 {'n_estimators': 200}
forest_regre = RandomForestClassifier(n_estimators = 200).fit(X_train,y_train)
y_pred4 = forest_regre.predict(X_val)
predicted_prob4 = forest_regre.predict_proba(X_val)
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


for clf, label in zip([clf1_regre, clf2_regre,clf3_regre,clf_regre,forest_regre],
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

filenameLR_RERG_NoChi2 = './models/WithoutChi2/LR_RERG_NoChi2.sav'
filenameSVM_RERG_NoChi2 = './models/WithoutChi2/SVM_RERG_NoChi2.sav'
filenameRF_RERG_NoChi2 = './models/WithoutChi2/RF_RERG_NoChi2.sav'
filenameGB_RERG_NoChi2 = './models/WithoutChi2/GB_RERG_NoChi2.sav'
filenameNB_RERG_NoChi2 = './models/WithoutChi2/NB_RERG_NoChi2.sav'
joblib.dump(clf3_regre, filenameLR_RERG_NoChi2)
joblib.dump(clf1_regre, filenameSVM_RERG_NoChi2)
joblib.dump(forest_regre, filenameRF_RERG_NoChi2)
joblib.dump(clf_regre, filenameGB_RERG_NoChi2)
joblib.dump(clf2_regre, filenameNB_RERG_NoChi2)