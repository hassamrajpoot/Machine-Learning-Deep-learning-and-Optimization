import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from DataGenerator import DataGenerator
import warnings
warnings.filterwarnings("ignore")

labels = pd.read_csv("../input/g2net-gravitational-wave-detection/training_labels.csv")
labels.head()

paths = glob("../input/g2net-gravitational-wave-detection/train/*/*/*/*")
ids = [path.split("/")[-1].split(".")[0] for path in paths]
path_df = pd.DataFrame({"path": paths, "id": ids})
train_df = pd.merge(left=labels, right=path_df, on="id")
train_df.head()

target_1 = train_df[train_df.target == 1]
target_0 = train_df[train_df.target == 0]

sample_submission = pd.read_csv('../input/g2net-gravitational-wave-detection/sample_submission.csv')
train_idx = labels['id'].values
y = labels['target'].values
test_idx = sample_submission['id'].values

train_idx, train_Valx = train_test_split(list(labels.index), test_size=0.33, random_state=2021)
test_idx = list(sample_submission.index)

train_generator = DataGenerator('/kaggle/input/g2net-gravitational-wave-detection/train/', train_idx, labels, 64)
val_generator = DataGenerator('/kaggle/input/g2net-gravitational-wave-detection/train/', train_Valx, labels, 64)
test_generator = DataGenerator('/kaggle/input/g2net-gravitational-wave-detection/test/', test_idx, sample_submission,
                               64)
model = Sequential()
model.add(Conv1D(64, input_shape=(3, 4096,), kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=2e-4), loss='binary_crossentropy', metrics=['acc'])

model.summary()

history = model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=1, workers=4)

predict = model.predict_generator(test_generator, verbose=1)
