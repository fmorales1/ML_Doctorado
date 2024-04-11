import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost

df = pd.read_csv("Base de datos.csv", header = 0)

X = df.loc[:,df.columns!="contraccion" ]
y = df.loc[:,"contraccion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 1)

X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 1)

xgb = xgboost.XGBClassifier()

parameters = {"nthreads": [1],
              "objetive" : ["binary:logistic"],
              "learning_rate":[0.05, 0.1],
              "n_estimators":[100,200]}

fitparams = {"early_stopping_rounds": 10,
              "eval_metric":"logloss",
              "eval_set":[(X_test,y_test)]}

clf = GridSearchCV(xgb, parameters, cv=3, scoring="accuracy")

