import cv2
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
import keras
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import pooling



_CORRECTION_NUM = 0.02

# ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']


def function(parent_dir):
    lines = []
    csv_file = './{}/driving_log.csv'.format(parent_dir)
    image_file_base = '/{}/IMG/'.format(parent_dir)

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    lines = iter(lines)
    _ = next(lines)

    return_images = []
    return_measurements = []
    for line in lines:
        # center
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = image_file_base + filename

        image = cv2.imread(current_path)
        return_images.append(image)
        return_images.append(cv2.flip(image, 1))

        measurement = float(line[3])
        return_measurements.append(measurement)
        return_measurements.append(measurement * -1.0)



        # left
        source_path = line[1]
        filename = source_path.split('/')[-1]
        current_path = image_file_base + filename


        image = cv2.imread(current_path)
        return_images.append(image)
        return_images.append(cv2.flip(image, 1))


        measurement = float(line[3])
        return_measurements.append(measurement + _CORRECTION_NUM)
        return_measurements.append((measurement+_CORRECTION_NUM)* -1.0)


        # right
        source_path = line[2]
        filename = source_path.split('/')[-1]
        current_path = image_file_base+ filename


        image = cv2.imread(current_path)
        return_images.append(image)
        return_images.append(cv2.flip(image, 1))


        measurement = float(line[3])
        return_measurements.append(measurement-_CORRECTION_NUM)
        return_measurements.append((measurement-_CORRECTION_NUM)* -1.0)
        return return_images, return_measurements


def train():
    parent_dirs = ['data', 'local-trained-data']

    all_images = []
    all_measurements = []
    for the_dir in parent_dirs:
        _images, _measurements = function(the_dir)
        all_images.extend(_images)
        all_measurements.extend(all_measurements)


    X_train = np.array(all_images)
    y_train = np.array(all_measurements)

    print(X_train.shape)
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

    model.compile(loss = 'mse', optimizer='adam')
    print('Printing...')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
    # increasing epoch doesn't seem to help
    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
    model.save('model.h5')
    print('Done')


train()
