# -*- coding: utf-8 -*-
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from confusion_matrix import plot_confusion_matrix

from datasetTools import getDataset
from config import slicesPath
from config import filesPerGenre
from config import validationRatio, testRatio
from config import sliceSize


# Training params
batch_size = 64
epochs = 4

# List genres
genres = os.listdir(slicesPath)
genres = [filename for filename in genres if os.path.isdir(
    slicesPath + filename)]
nbClasses = len(genres)

# Create or load new dataset
train_X, train_y, validation_X, validation_y = getDataset(
    filesPerGenre, genres, sliceSize, validationRatio, testRatio, mode="train")

test_X, test_y = getDataset(
    filesPerGenre, genres, sliceSize, validationRatio, testRatio, mode="test")


def restrict_to_top_6_genres_single(X, y, top6_idxs):
    keep = np.array([y.argmax(1) == i for i in top6_idxs]).sum(0).astype(bool)
    return X[keep], y[keep]


def restrict_to_top_6_genres(train_X, train_y, validation_X, validation_y, test_X=None, test_y=None):
    y = train_y
    genre_counts = [sum(y.argmax(1) == i) for i in range(y.shape[1])]
    top6_idxs = np.argsort(genre_counts)[-6:]
    train_X, train_y = restrict_to_top_6_genres_single(
        train_X, train_y, top6_idxs)
    validation_X, validation_y = restrict_to_top_6_genres_single(
        validation_X, validation_y, top6_idxs)
    if test_X is not None:
        test_X, test_y = restrict_to_top_6_genres_single(
            test_X, test_y, top6_idxs)
        return train_X, train_y, validation_X, validation_y, test_X, test_y, top6_idxs
    return train_X, train_y, validation_X, validation_y, top6_idxs


# Remove all but the top 6 genres
train_X, train_y, validation_X, validation_y, test_X, test_y, top6_idxs = \
    restrict_to_top_6_genres(
        train_X, train_y, validation_X, validation_y, test_X, test_y)

num_classes = train_y.shape[1]

# input image dimensions
img_rows, img_cols = train_X.shape[2], train_X.shape[1]

train_X = train_X.astype('float32')
validation_X = validation_X.astype('float32')

print('train_X shape:', train_X.shape)
print(train_X.shape[0], 'train samples')
print(validation_X.shape[0], 'test samples')

# Define and compile keras model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='elu',
                 input_shape=[img_cols, img_rows, 1]))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(1024, kernel_size=(3, 3), activation='elu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(validation_X, validation_y))
score = model.evaluate(validation_X, validation_y, verbose=0)
print('Val loss:', score[0])
print('Val accuracy:', score[1])

score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

preds = model.predict(test_X)
plt.figure(figsize=[12] * 2)
plot_confusion_matrix(test_y.argmax(1), preds.argmax(
    1), [genres[i] for i in top6_idxs], normalize=False)
