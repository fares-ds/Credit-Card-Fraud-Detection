import pandas as pd
# import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

from xgboost import XGBClassifier

epochs = 10
df = pd.read_csv('data/creditcard.csv')


def process_data(data):
    """
    Data Pre-processing
    :returns processed data split into train and test
    """
    # Train Test split
    X = data.drop('Class', axis=1)
    y = data.Class

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scaling the data
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    return X_train, X_test, y_train, y_test


def evaluate(label, prediction, train=True):
    if train:
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.4f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(label, prediction) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(label, prediction) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(label, prediction) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, prediction)}\n")

    elif train == False:
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.4f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(label, prediction) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(label, prediction) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(label, prediction) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(label, prediction)}\n")


X_train, X_test, y_train, y_test = process_data(df)
print(f"TRAINING: X_train: {X_train.shape}, y_train: {y_train.shape}\n{'_' * 55}")
print(f"TESTING: X_test: {X_test.shape}, y_test: {y_test.shape}")

# xgboost Model
xgb_clf = XGBClassifier()

# Hyper-parameter Tuning
params = {
    'n_estimators': [50, 100, 500, 1000],
    'learning_rate': [0.1, 0.01, 0.001],
    'base_score': [0.1, 0.25, 0.5, 0.75, 0.99]
}

xgb_cv = GridSearchCV(estimator=xgb_clf, param_grid=params, scoring='f1', n_jobs=-1, cv=5)
# Fit the model
xgb_cv.fit(X_train, y_train)

print(f'GRID SEARCH BEST ESTIMATOR:\n{xgb_cv.best_estimator_}')

y_train_pred = xgb_cv.predict(X_train)
y_test_pred = xgb_cv.predict(X_test)

evaluate(y_train, y_train_pred.round(), train=True)
evaluate(y_test, y_test_pred.round(), train=False)
