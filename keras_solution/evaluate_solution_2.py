from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from PIL import Image
import os
from keras import applications
import tensorflow as tf

# dimensions of our images.
img_width, img_height = 150, 150
batch_size = 1
top_model_weights_path = './models/bottleneck_fc_model.h5'
bottleneck_features ='./models/bottleneck_features_test.npy'
test_data_dir = './data/test'

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes=None,
    class_mode=None,
    shuffle=False)
dirs = os.listdir(test_data_dir)
paths = []
for dir in dirs:
    if os.path.isdir(os.path.join(test_data_dir, dir)) and not dir.startswith("."):
        paths.extend(os.listdir(os.path.join(test_data_dir, dir)))
paths = sorted(paths)
if "test" in test_data_dir:
    paths = [os.path.join("unlabeled", p) for p in paths]
else:
    paths = [os.path.join("cats", p) if "cat" in p else os.path.join("dogs", p) for p in paths]
paths = [os.path.join(test_data_dir, p) for p in paths]
# build the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')

if not tf.gfile.Exists(bottleneck_features):
    bottleneck_features_test = model.predict_generator(
        test_generator, 12500)
    np.save(bottleneck_features, bottleneck_features_test)
else:
    bottleneck_features_test = np.load(bottleneck_features)

model = Sequential()
model.add(Flatten(input_shape=bottleneck_features_test.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.load_weights(top_model_weights_path)

submission = []
for pred_count in range(bottleneck_features_test.shape[0]):
    if pred_count == 12500:
        break
    batch = np.expand_dims(bottleneck_features_test[pred_count], axis=0)
    # img = Image.open(paths[pred_count])
    predict = model.predict_on_batch(batch)
    # img.show()
    # print(1 - predict[0][0])
    id, ext = os.path.splitext(paths[pred_count])
    submission.append([id, 1 - predict[0][0]])

import csv
with open('submission_2.csv', 'w') as csvfile:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for sub in submission:
        writer.writerow({'id': sub[0], 'label': sub[1]})
