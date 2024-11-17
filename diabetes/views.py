from django.shortcuts import render,redirect
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Create your views here.

def home(request):
    return render(request, 'index.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    df= pd.read_csv('diabetes.csv')
    X=df.drop('Outcome',axis=1)
    y=df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)
    
    # Define base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
    ]

    # Meta-model
    meta_model = LogisticRegression()

    # Stacking ensemble
    stack_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

    # Fit the model
    stack_model.fit(X_train, y_train)

    # # Evaluate
    # train_accuracy = stack_model.score(X_train, y_train)
    # val_accuracy = stack_model.score(X_test, y_test)

    # print("Training Accuracy (Stacking):", train_accuracy)
    # print("Validation Accuracy (Stacking):", val_accuracy)
    
    val1=float(request.GET['n1'])
    val2=float(request.GET['n2'])
    val3=float(request.GET['n3'])
    val4=float(request.GET['n4'])
    val5=float(request.GET['n5'])
    val6=float(request.GET['n6'])
    val7=float(request.GET['n7'])
    val8=float(request.GET['n8'])

    pred = stack_model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])
    
    result1 = ""
    if  pred==[1]:
        result1="Positive"
    else:
        result1="Neagtive"
            
    
    
    return render(request,'predict.html',{"result2":result1})

