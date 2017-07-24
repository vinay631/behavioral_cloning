import pandas as pd
import seaborn as sns
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt
import glob
import os
from sklearn.utils import shuffle
from utils import augment, preprocess_img
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Cropping2D, Convolution2D, Conv2D
from keras.layers.advanced_activations import ELU
from keras.layers.noise import GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

def generate_data(image_paths, angles, batch_size=20, is_validation=False):
  '''A generator method to create training or validation by augmenting and
  preprocessing images.
  '''

  X = []
  y = []
  image_paths, angles = shuffle(image_paths, angles)

  while True:
    for i in range(batch_size):
      c, l, r = image_paths[i]
      angle = angles[i]
      
      if not is_validation:
        img, angle = augment(c, l, r, angle)
      else:
        img = load(c)
      img = preprocess_img(img)
      X.append(img)
      y.append(angle)
    yield (np.array(X), np.array(y)) 

def build_model():
  '''This model is base on NVIDIA model with slight modification.'''

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(66, 200, 3)))
  model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
  model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
  model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
  model.add(Conv2D(64, 3, 3, activation='elu'))
  model.add(Conv2D(64, 3, 3, activation='elu'))
  model.add(Dropout(.5))
  model.add(Flatten())
  model.add(Dense(100, activation='elu'))
  model.add(Dense(50, activation='elu'))
  model.add(Dense(10, activation='elu'))
  model.add(Dense(1))
  model.summary() 

  return model

def train(image_paths, angles, num_epochs=10):
  '''Trains the model on images and angles.'''

  BATCH_SIZE = 32
  X_train, X_valid, y_train, y_valid = train_test_split(image_paths, angles, test_size=0.2, random_state=0)

  checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

  model = build_model()

  model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))

  model.fit_generator(generate_data(X_train, y_train, batch_size=BATCH_SIZE),
                      samples_per_epoch=16000,
                      nb_epoch=num_epochs,
                      max_q_size=1,
                      validation_data=generate_data(X_valid, y_valid, batch_size=BATCH_SIZE, is_validation=True),
                      nb_val_samples=len(X_valid),
                      callbacks=[checkpoint],
                      verbose=1)

  print('Training Complete!') 

  return model


def save_model(model):
  '''Save the model to json file and the weights to h5 file.'''
  model.save_weights('./model.h5')
  json_string = model.to_json()
  with open('./model.json', 'w') as f:
    f.write(json_string)

def read_data(data_dir, log_file_name='driving_log.csv'):
  '''Reads the image paths and angle associated with those images.'''
  
  driving_log_df = pd.read_csv(os.path.join(data_dir, log_file_name), header=None)
  driving_log_df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

  image_paths = list(zip(driving_log_df.center.tolist(),
                         driving_log_df.left.tolist(),
                         driving_log_df.right.tolist()))

  angles = driving_log_df.steering.tolist()

  return image_paths, angles

def main():
  '''Read images, create and train model.'''

  parser = argparse.ArgumentParser(description='Behavioral Cloning')
  parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')  
  parser.add_argument('-e', help='number of epochs', dest='num_epochs', type=int, default='10')

  args = parser.parse_args()
  
  data_dir = args.data_dir
  num_epochs = args.num_epochs

  image_paths, angles = read_data(data_dir)
  
  model = train(image_paths, angles, num_epochs=num_epochs)

  save_model(model)
  

if __name__=='__main__':
  '''The script should be run as: python model.py -d data_dir -e num_epochs.'''
  main()
