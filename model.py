import cv2
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
import keras
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import pooling, Dropout
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



_CORRECTION_NUM = 0.02


def _get_data():
    parent_dirs = [
        # 'data',
        # 'local-trained-data',
        'local-trained-data-opposite-direction',
        'local-trained-data-original-direction',
        # 'local-trained-data-off-tracks-new',
        'local-trained-data-along-curves',
    ]


    lines = []
    for p in parent_dirs:
        csv_file = './{}/driving_log.csv'.format(p)
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

    return train_test_split(lines, test_size=0.2)


def _generator(samples, batch_size=32):
    # csv data format:
    # ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        print('hi loop')
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                directory = batch_sample[0].split('/')[-3]
                image_path = batch_sample[0].split('/')[-1]
                name = './{}/IMG/{}'.format(directory, image_path)

                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)



def train():
    train_samples, validation_samples = _get_data()
    train_generator = _generator(train_samples, batch_size=32)
    validation_generator = _generator(validation_samples, batch_size=32)


    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.add(Dropout(0.1))


    model.compile(loss = 'mse', optimizer='adam')
    print('Printing...')
    model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        verbose=1,
        nb_epoch=3,
    )
    model.save('model.h5')
    print('Done')


train()
