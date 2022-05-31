# StrokePrediction_EDA_ML

## Abstract
Stroke is quite harmful for an organism, especially for the older. As a result, there is a need to be able to 
predict the probability of risk in a direct and easy way for a stroke to occur. Using big data through 
sensors, we can now record data for each patient, to perform analysis. The specific research based on 
medical data, focuses on creating machine learning models to predict the likelihood of a stroke 
occurring. There is also a Mockup from the creation of the corresponding application and the analysis 
results from EDA procedure.

## Dataset
The Data, for this project, were collected from Kaggle in csv format and they are annotated.

## Jupyter Notebook

In order to see the whole process and clarrifications during training/validation and test, see the "Stroke Prediction EDA and ML prediction.ipynb" file and the source code under the scr folder in this Repository.
In order to see all of the plots in notebook produced by pyplot you can [clik here](https://nbviewer.org/github/icsd13152/StrokePrediction_EDA_ML/blob/main/StrokePrediction/src/Stroke%20Prediction%20EDA%20and%20ML%20prediction.ipynb). In this link you can see the whole notebook too.

## Application
Dash package and framework is used to create this application and the user interface. A user interface 
mockup is shown below. In the below image we can see some predicted results in a dashboard. The 
dashboard is an easy-to-use tool. In this way, we are able to have a useful and simple user interface. The 
group of Risk level (Low, Medium, High) are designed and created based on the optimal threshold of 
ROC curves from the corresponding Classifier. Due to the fact that we are interesting about the Stroke, 
classifier predicts only the probability of having Stroke, based on the patient profile that user will give as 
input.

![Application](https://github.com/icsd13152/StrokePrediction_EDA_ML/blob/main/StrokePrediction/mockup/mediumrisk.PNG?raw=true)