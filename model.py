import csv
import cv2
import numpy as np

training_dir = 'data'

print("Reading the driving log...")
lines = []
with open(training_dir + '/driving_log.csv') as d_log:
   reader = csv.reader(d_log)
   for line in reader:
      lines.append(line)

print("Building the images...")
model_images = []
model_measurements = []
for line in lines:
   line = [part.strip() for part in line]
   source_path = line[0]
   filename = source_path.split('/')[-1]
   current_path = training_dir + '/IMG/' + filename
   image = cv2.imread(current_path)
   model_images.append(image)

   measurement = float(line[3])
   model_measurements.append(measurement)

images = []
measurements = []
for model_image, model_measurement in zip(model_images, model_measurements):
   images.append(model_image)
   measurements.append(model_measurement)

   images.append(np.fliplr(model_image))
   measurements.append(-model_measurement)
   

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D

def preprocess(model):
   model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
   model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))

def lenet(model):
   model.add(Convolution2D(6,5,5, activation = 'relu'))
   model.add(MaxPooling2D())
   model.add(Convolution2D(6,5,5, activation = 'relu'))
   model.add(MaxPooling2D())
   model.add(Flatten())
   model.add(Dense(120))
   model.add(Dense(84))


model = Sequential()
preprocess(model)

lenet(model)

model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')

print("Training the network...")
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

model.save('model.h5')

