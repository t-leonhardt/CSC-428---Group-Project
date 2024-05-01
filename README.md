# CSC-428---Group-Project

## Dataset used:
    Train_Test_IoT_GPS_Tracker
        Subset of TON-IoT
    6 attributes (latitude, longitude, time, date, type, and label)
        Type describes interaction (Normal,  DDOS, Backdoor, Injection, Password, Ransomware, Scanning, and XSS)
    38,960 instances

## Machine Learning models:
    K-Nearest Neighbor
    Decision tree
    Random Forest
    all models use:
        80/20 split
        10 fold cross validation

## Results:
    K-Nearest Neighbor (KNN):
        Accuracy: 97.70%
        Precision: 97.13%
        Recall: 94.39%
        F1 score: 95.59%

    Decision Tree (DT):
        Accuracy: 99.97%
        Precision: 99.98%
        Recall: 99.88%
        F1 score: 99.93%

    Random Forest (RF):
        Accuracy: 99.99%
        Precision: 99.99%
        Recall: 99.89%
        F1 score: 99.94%

## Additional Findings:
 Upon reflection, I recognized that including the "date" and "time" attributes might have provided the algorithm with overly simplistic decision-making criteria, resulting in accuracy rates in the mid-80 percentile range. Consequently, we concluded that only the "longitude," "latitude," and "type" attributes should be considered when training a model on this dataset. By excluding the "date," "time," and "label" attributes, we can prevent the model from relying on overly simplistic decision boundaries and achieve more robust and realistic results.