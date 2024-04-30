from utils.datahandler import DataHandler
from interpret.glassbox import LogisticRegression, ClassificationTree
from interpret import show
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import pandas as pd


seed = 2021

datahandler = DataHandler()

datahandler.load_data()
datahandler.standarization()

df = datahandler.get_data()
X_train, X_test, y_train, y_test = datahandler.get_data_split(seed)

lr = LogisticRegression(random_state=seed, feature_names=X_train.columns, penalty='l1', solver='liblinear')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(f"F1 Score {f1_score(y_test, y_pred)}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# show(lr.explain_global())

ct = ClassificationTree()
ct.fit(X_test, y_test)

y_pred = ct.predict(X_test)

print(f"F1 Score {f1_score(y_test, y_pred)}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")