# Created by Henry Bowman Hill
# DTC Classifier to model flu risk based off of demographic features

import pandas as pd
import numpy as np

# Read in the csv as a dfframe
input_file = "nhis_clean.csv"
GFdf = pd.read_csv(input_file, header = 0)
del GFdf[GFdf.columns[0]]
print(GFdf['VACFLUSH12M'].value_counts())

#DTC classifier
x = GFdf.drop(columns="FLUPNEUYR")
y = GFdf["FLUPNEUYR"]

feature_names = x.columns
labels = y.unique()

#split the dataset
from sklearn.model_selection import train_test_split
X_train, test_x, y_train, test_lab = train_test_split(x,y,test_size = 0.4,random_state = 42)

# Train the model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 10, random_state = 42)

clf.fit(X_train, y_train)
test_pred_decision_tree = clf.predict(test_x)

# main function to call the classifier
def main(age,nchild,race,regionbr,health,bmi,vacc):
    #binning age from user
    if age >= 60:
        age = 1
    elif age <= 18:
        age = 1
    else:
        age = 0
        
    #call classifier
    UserIn = {'AGE': [age], 'NCHILD': [nchild], 'RACENEW':[race],'REGIONBR':[regionbr],'HEALTH':[health],'BMICAT':[bmi],"VACFLUSH12M":[vacc]}
    Udf = pd.DataFrame(data = UserIn)
    pred = clf.predict(Udf)
    return pred

test = main(65,1,100,1,1,4,1)
print(test)
#plotting heatmap to check model accuracy
def plotHeatMap():
    from sklearn import metrics
    import seaborn as sns
    import matplotlib.pyplot as plt

    confusion_matrix = metrics.confusion_matrix(test_lab, test_pred_decision_tree)

    matrix_df = pd.DataFrame(confusion_matrix)

    ax = plt.axes()

    sns.set(font_scale=1.3)

    plt.figure(figsize=(10,7))

    sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")

    ax.set_title('Confusion Matrix - Decision Tree')

    ax.set_xlabel("Predicted label", fontsize = 15)

    ax.set_ylabel("True Label", fontsize = 15)

    ax.set_yticklabels(list(labels), rotation = 0)

    plt.show()
    
