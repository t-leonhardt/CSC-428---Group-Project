from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict

data=pd.read_csv("/kaggle/input/dataset/Train_Test_IoT_GPS_Tracker.csv")

class_labels=data['type'].unique()

le=LabelEncoder()

data['type']=le.fit_transform(data['type'])

date=le.fit_transform(data['date'])
time=le.fit_transform(data['time'])

data.drop("date", axis=1, inplace=True)
data["date"]=date
data.drop('time', axis=1, inplace=True)
data['time']=time
target_column=['type']
y=data[target_column]
feature_columns=['date', 'time', 'latitude', 'longitude']
X=data[feature_columns]
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

knn_clf=KNeighborsClassifier()
knn_cv_scores=cross_val_score(knn_clf, X_train, y_train.values.ravel(), cv=10)

knn_clf.fit(X_train, y_train.values.ravel())

knn_cv_pred=cross_val_predict(knn_clf, X_test,y_test.values.ravel(), cv=10)

knn_accuracy=accuracy_score(y_test, knn_cv_pred)
knn_precision=precision_score(y_test, knn_cv_pred, average='macro')
knn_recall=recall_score(y_test, knn_cv_pred, average='macro')
knn_f1=f1_score(y_test, knn_cv_pred, average='macro')

print("K-Nearest Neighbors Metrics ")
print("Accuracy: ", knn_accuracy)
print("Precision: ", knn_precision)
print("Recall: ", knn_recall)
print("F1 score: ", knn_f1)


conf_matrix=confusion_matrix(y_test, knn_cv_pred)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()