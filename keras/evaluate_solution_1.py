from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from PIL import Image
import os

# dimensions of our images.
img_width, img_height = 150, 150
batch_size = 1

test_data_dir = './data/test'

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.load_weights("./models/first_try.h5")

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes=None,
    class_mode=None,
    shuffle=False)

paths = os.listdir('./data/test/unlabeled')
submission = []
for pred_count, batch in enumerate(test_generator):
    # img = np.asarray(batch[0], np.uint8)
    # img = Image.fromarray(img)
    predict = model.predict_on_batch(batch)
    # img.show()
    id = os.path.basename(paths[pred_count])
    submission.append([id, 1 - predict[0][0]])

import csv
with open('submission.csv', 'w') as csvfile:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for sub in submission:
        writer.writerow({'id': sub[0], 'label': sub[1]})
