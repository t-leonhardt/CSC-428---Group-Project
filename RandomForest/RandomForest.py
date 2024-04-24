# CSC 428 - RandomForest 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("/kaggle/input/train-test-iot-gps-tracker/Train_Test_IoT_GPS_Tracker.csv")

class_labels = data['type'].unique()

le = LabelEncoder()
data['type'] = le.fit_transform(data['type']) 
date = le.fit_transform(data['date'])
time = le.fit_transform(data['time'])

data.drop("date", axis=1, inplace=True)
data["date"] = date
data.drop('time', axis = 1, inplace = True)
data['time'] = time

y = data['type'].values.flatten()
feature_columns = ['date', 'time', 'latitude', 'longitude']
X = data[feature_columns]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

rf_clf = RandomForestClassifier()
rf_cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=10, scoring='accuracy')

rf_cv_pred = cross_val_predict(rf_clf, X_test, y_test, cv=10)

rf_accuracy = accuracy_score(y_test,rf_cv_pred)
rf_precision = precision_score(y_test, rf_cv_pred, average='macro')
rf_recall = recall_score(y_test, rf_cv_pred, average = 'macro')
rf_f1 = f1_score(y_test, rf_cv_pred, average = 'macro')

print("Random Forest Metrics ")
print("Accuracy: ", rf_accuracy)
print("Precision: ", rf_precision)
print("Recall: ", rf_recall)
print("F1 score:", rf_f1)


conf_matrix = confusion_matrix(y_test, rf_cv_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()