
from patient import Patient
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report

PATH_TO_FOLDER = "./images"
IMAGE_FORMAT = "{group}_wc1_{patient_number}_{slice_number}.jpg"

INDEX_OF_AD_PATIENTS = {"first": 1, "last": 1497}
INDEX_OF_CN_PATIENTS = {"first": 1501, "last": 3000}
INDEX_OF_EMCI_PATIENTS = {"first": 3001, "last": 3388}
INDEX_OF_LMCI_PATIENTS = {"first": 70, "last": 1500}
INDEX_OF_MCI_PATIENTS = {"first": 1501, "last": 3000}
INDEX_OF_SMC_PATIENTS = {"first": 3001, "last": 3662}

SLICE = 55

matrix = []
group_array = []

"""
new_patient = Patient(1, "ad", normalize=False)
hey = new_patient.get_single_image_array(55)
print(hey.shape)

"""
"""
print("loading ad patients...")
for i in range(INDEX_OF_AD_PATIENTS["first"], INDEX_OF_AD_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "ad", normalize=False)
        matrix = []
        matrix.append(new_patient.get_single_image_array(SLICE))
        group_array.append(0)
    except:
        print(f"Couldn't load patient {i}")
        continue

print("loading cn patients...")
for i in range(INDEX_OF_CN_PATIENTS["first"], INDEX_OF_CN_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "cn", normalize=False)
        matrix = []
        matrix.append(new_patient.get_single_image_array(SLICE))
        group_array.append(1)
    except:
        print(f"Couldn't load patient {i}")
        continue

print("loading mci patients...")
for i in range(INDEX_OF_MCI_PATIENTS["first"], INDEX_OF_MCI_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "mci", normalize=False)
        matrix = []
        matrix.append(new_patient.get_single_image_array(SLICE))
        group_array.append(2)
    except:
        print(f"Couldn't load patient {i}")
        continue

print("loading emci patients...")
for i in range(INDEX_OF_EMCI_PATIENTS["first"], INDEX_OF_EMCI_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "emci", normalize=False)
        matrix = []
        matrix.append(new_patient.get_single_image_array(SLICE))
        group_array.append(3)
    except:
        print(f"Couldn't load patient {i}")
        continue

print("loading lmci patients...")
for i in range(INDEX_OF_LMCI_PATIENTS["first"], INDEX_OF_LMCI_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "lmci", normalize=False)
        matrix = []
        matrix.append(new_patient.get_single_image_array(SLICE))
        group_array.append(4)
    except:
        print(f"Couldn't load patient {i}")
        continue

print("loading smc patients...")
for i in range(INDEX_OF_SMC_PATIENTS["first"], INDEX_OF_SMC_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "smc", normalize=False)
        matrix = []
        matrix.append(new_patient.get_single_image_array(SLICE))
        group_array.append(5)
    except:
        print(f"Couldn't load patient {i}")
        continue

#np.save("array_cnn.npy", matrix)
np.save("group_cnn.npy", group_array)
"""

X = np.load("array_cnn.npy")
Y = np.load("group_cnn.npy")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=11, stratify=Y)

"""
#Normal Neuronal Network
ann = models.Sequential([
        layers.Flatten(input_shape=(224,224,3)),
        layers.Dense(50, activation='relu'),
        layers.Dense(6, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, Y_train, epochs=5)

y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(Y_test, y_pred_classes))

"""

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, Y_train, epochs=10)
cnn.evaluate(X_test, Y_test)
y_pred = cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(Y_test, y_pred_classes))