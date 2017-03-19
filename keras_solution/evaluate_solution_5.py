import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k
import csv
from keras.models import model_from_json
# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# hyper parameters for model
based_model_last_block_layer_number = 126  # value is based on based model selected.
img_width, img_height = 229, 229  # change based on the shape/structure of your images
batch_size = 1  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
final_model_weights_path = './models/model_weights_inception.h5'
model_path = './models/model_inception.json'
nb_train_samples = 20000
nb_validation_samples = 5000
test_data_dir = './data/test'
results_name = 'submission_5.csv'

json_file = open(model_path, 'r')

loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(final_model_weights_path)

# Read Data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  shuffle=False)

# Calculate class posteriors probabilities
y_probabilities = model.predict_generator(test_generator,
                                          steps=12500 // batch_size)
y_probabilities = y_probabilities[:][0]
y_probabilities = 1 - y_probabilities
# Calculate class labels

filenames = [filename.split('/')[1] for filename in test_generator.filenames]
ids = [filename.split('.')[0] for filename in filenames]

submission = list(zip(ids, y_probabilities))
submission = sorted(submission, key=submission[0])

# save results as a csv file in the specified results directory
with open(results_name, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(('id', 'label'))
    writer.writerows(submission)
