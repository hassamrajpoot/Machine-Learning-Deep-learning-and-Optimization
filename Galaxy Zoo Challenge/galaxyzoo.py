import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, GlobalMaxPooling2D
from keras import backend as K
from skimage.transform import resize
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

df = pd.read_csv('../input/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.zip')
df.head()

#!unzip ../input/galaxy-zoo-the-galaxy-challenge/images_training_rev1.zip

df = df.rename(columns={'GalaxyID':'GalaxyImage'})
df.head()

df['GalaxyImage'] = ["/kaggle/working/images_training_rev1/{}.jpg".format(img) for img in list(df['GalaxyImage'].to_numpy())]
df.head()

img = mpimg.imread(df['GalaxyImage'][0])
imgplot = plt.imshow(img)
plt.show()

df_train, df_test = train_test_split(df, test_size=.2)

df_train.head()

orignal_dimensions = (424,424)
crop_dimesions = (256,256)
final_img_dimensions = (64,64)

def process_image(path, x,y, shape, crop_dimensions):
    img = plt.imread(path)
    img = img[x:x+crop_dimesions[0], y:y+crop_dimesions[1]]
    img = resize(img, shape)
    img = img/255.
    return img
    
def process_all_images(dataframe, shape=final_img_dimensions, crop_size=crop_dimesions):
    x1 = (orignal_dimensions[0]-crop_dimesions[0])//2
    y1 = (orignal_dimensions[1]-crop_dimesions[1])//2
    ids = dataframe['GalaxyImage'].values
    y_batch = dataframe.iloc[:,1:].values
    x_batch = []
    for i in tqdm(ids):
        x = process_image(i, x1,y1, shape=shape, crop_size=crop_dimesions)
        x_batch.append(x)
    x_batch = np.array(x_batch)
    return x_batch, y_batch
X_train, y_train = process_all_images(df_train)
X_test, y_test = process_all_images(df_test)

model = Sequential()
model.add(Conv2D(512, (3, 3), input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 3)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(37))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=[lambda y_true , y_pred : K.sqrt(K.mean(K.square(y_pred - y_true)))])
model.summary()

batch_size = 64
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

model.evaluate(X_test,y_test)
