# Scraping (Most likely won't be needed)
import bs4 as bs

# Visualization/Organization
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import cv2 as cv

# Analysis
import nltk as ntlk
import keras
import tensorflow as tf

# Predictiction
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# for file management if needed
import os  

def main():

# Test for tensorflow
    cifar = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(32, 32, 3),
        classes=100,)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=64)


if __name__ == "__main__":
    main()