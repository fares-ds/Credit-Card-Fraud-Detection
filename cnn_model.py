import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
# from keras.optimizers import Adam

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

    # For cnn, we need 2-D array
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    return X_train, X_test, y_train, y_test


def cnn_model(X_train):
    model = Sequential()
    model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv1D(64, 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))
    return model


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

# CNN Model
model = cnn_model(X_train)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.AUC()])
r = model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              batch_size=50,
              epochs=epochs)

training_score = model.evaluate(X_train, y_train)
testing_score = model.evaluate(X_test, y_test)

print(f"TRAINING SCORE: {training_score}")
print(f"TESTING SCORE: {testing_score}")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

evaluate(y_train, y_train_pred.round(), train=True)
evaluate(y_test, y_test_pred.round(), train=False)
