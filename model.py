import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import gc
import random
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Lambda, Dropout, Cropping2D, Activation, Merge, Input, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.core import *
from keras.models import Model

# the following helper class for loading the data using generators
class DataSet:
    # data split
    lines_train = None
    lines_test = None

    # generators
    gen_train = None
    gen_test = None

    # data size
    train_size = None
    test_size = None

    image_shape = None

    # base location of data on disk
    base_path = None

    batch_size = None

    def __init__(self, data_file, batch_size=64):
        self.batch_size = batch_size
        self.base_path = data_file.rsplit("/", 1)[0]
        lines = []
        with open(data_file) as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                lines.append(line)
        lines = lines[1:]
        self.lines_train, self.lines_test = train_test_split(lines, test_size=0.2)
        self.train_size = len(self.lines_train)
        self.test_size = len(self.lines_test)
        self.image_shape = cv2.imread(self.base_path + "/" + lines[0][0].strip()).shape

    # generator
    def __generate(self, lines):
        while True:
            shuffle(lines)
            for offset in range(0, len(lines), self.batch_size):
                batch = lines[offset: offset + self.batch_size]

                views = []#the images from camera
                steers = []#corresponding camera angle
                for line in batch:
                    #images from center, left and right camera per line in csv file
                    view = [cv2.imread(self.base_path + "/" + (line[i].strip())) for i in range(3)]
                    #sterring angles, with a variation of 0.25 for the left and right camera images
                    steer = [float(line[3]), 0.25 + float(line[3]), float(line[3]) - 0.25]

                    #flip to produce augmented images
                    for img in list(view):
                        view.append(np.fliplr(img))

                    #flip corresponding camera angles
                    for angle in list(steer):
                        steer.append(-angle)

                    #now we have 6 data points from a single csv record

                    #shuffle by selecting random indices
                    index = np.zeros(len(view))
                    if len(views) > 0:
                        index = [random.randrange(0, len(views) + i) for i in range(len(index))]
                    for i in range(len(index)):
                        views.insert(int(index[i]), view[i])
                        steers.insert(int(index[i]), steer[i])

                X = np.array(views)
                y = np.array(steers)
                gc.collect()
                yield X, y

    # training samples generator
    def for_training(self):
        if self.gen_train is None:
            self.gen_train = self.__generate(self.lines_train)
        return self.gen_train

    # testing samples generator
    def for_testing(self):
        if self.gen_test is None:
            self.gen_test = self.__generate(self.lines_test)
        return self.gen_test

    # end of class


dataset = DataSet("./data/driving_log.csv")


# the nvidia cnn architecture modified to train the model
def nvidia_sequential():
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=dataset.image_shape))
    model.add(Lambda(lambda x: (x / 127.5) - 1))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


# the functional api implementation for the nvidia architecture
def nvidia_functional():
    imagein = Input(shape=dataset.image_shape)
    layer = Cropping2D(cropping=((70, 20), (0, 0)))(imagein)
    layer = Lambda(lambda x: (x / 127.5) - 1)(layer)
    layer = Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2))(layer)
    layer = Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2))(layer)
    layer = Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2))(layer)
    layer = Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1))(layer)
    layer = Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1))(layer)
    layer = Flatten()(layer)
    layer = Dense(100, activation="elu")(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(50, activation="elu")(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(10)(layer)
    steerout = Dense(1)(layer)
    model = Model(inputs=imagein, outputs=steerout)
    return model


#comma.ai architecture
def comma_ai():
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=dataset.image_shape))
    model.add(Lambda(lambda x: x / 127.5 - 1.0))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model


model = nvidia_sequential()
model.compile(optimizer="adam", loss="mse")
model.fit_generator(generator=dataset.for_training(),
                    steps_per_epoch=dataset.train_size / dataset.batch_size,
                    validation_data=dataset.for_testing(),
                    validation_steps=dataset.test_size / dataset.batch_size,
                    epochs=10)

model.save('./model.h5')