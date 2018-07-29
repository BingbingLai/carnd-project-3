import cv2 
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
import keras
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import pooling

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        

correction_num = 0.02
images = []
measurements = []

# ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

lines = iter(lines)
_ = next(lines)

for line in lines:
    # center
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    
    image = cv2.imread(current_path)
    images.append(image)
    images.append(cv2.flip(image, 1))
    
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement * -1.0)
    

    
    # left
source_path = line[1]
filename = source_path.split('/')[-1]
current_path = './data/IMG/' + filename


image = cv2.imread(current_path)
images.append(image)
images.append(cv2.flip(image, 1))


measurement = float(line[3])
measurements.append(measurement + correction_num)
measurements.append((measurement+correction_num)* -1.0)


# right
source_path = line[2]
filename = source_path.split('/')[-1]
current_path = './data/IMG/' + filename


image = cv2.imread(current_path)
images.append(image)
images.append(cv2.flip(image, 1))


measurement = float(line[3])
measurements.append(measurement-correction_num)
measurements.append((measurement-correction_num)* -1.0)

    
    
X_train = np.array(images)
y_train = np.array(measurements)
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
