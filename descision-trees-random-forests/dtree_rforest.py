import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

loans = pd.read_csv("loan_data.csv")

'''
Different Plots for Visualization

loans[loans["credit.policy"] == 1]["fico"].hist(
    bins=35, color="blue", label="Credit Policy = 1", alpha=0.6)
loans[loans["credit.policy"] == 0]["fico"].hist(
    bins=35, color="red", label="Credit Policy = 0", alpha=0.6)

loans[loans["not.fully.paid"] == 1]["fico"].hist(
    bins=35, color="blue", label="Not Fully Paid = 1", alpha=0.6)
loans[loans["not.fully.paid"] == 0]["fico"].hist(
    bins=35, color="red", label="Not Fully Paid = 0", alpha=0.6)

plt.figure(figsize=(11, 7))
sn.countplot(x="purpose", hue="not.fully.paid", data=loans, palette="Set1")

sn.jointplot(x="fico", y="int.rate", data=loans, color="purple")

sn.lmplot(y="int.rate", x="fico", data=loans, hue="credit.policy",
          col="not.fully.paid", palette="Set1")
plt.show()

plt.legend()
plt.xlabel("FICO")
'''

cat_feats = ["purpose"]

final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)


X = final_data.drop("not.fully.paid", axis=1)
y = final_data["not.fully.paid"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)


dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

## Predictions and Evaluations
predictions = dtree.predict(X_test)

print("Classification Report: \n")
print(classification_report(y_test, predictions))
print("Confusion Matrix: \n")
print(confusion_matrix(y_test, predictions))

# Random Forest

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)

## Predictions and Evaluations
predictions2 = rfc.predict(X_test)

print("Classification Report: \n")
print(classification_report(y_test, predictions))
print("Confusion Matrix: \n")
print(confusion_matrix(y_test, predictions))
