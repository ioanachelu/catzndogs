from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, ZeroPadding2D, Convolution2D, MaxPooling2D
import numpy as np
import os
import h5py
from keras.utils.conv_utils import convert_kernel

# path to the model weights files.
top_model_weights_path = './models/bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 20000
nb_validation_samples = 5000
epochs = 50
batch_size = 16

# build the VGG16 network
model = Sequential()
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(img_width, img_height, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

model.add(Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
model.add(Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

model.add(Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
model.add(Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
model.add(Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
# assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
# f = h5py.File(weights_path)
# for k in range(f.attrs['nb_layers']):
#     if k >= len(model.layers):
#         # we don't look at the last (fully-connected) layers in the savefile
#         break
#     g = f['layer_{}'.format(k)]
#     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
#     if len(weights) > 0:
#         converted_w = convert_kernel(weights)
#         model.layers[k].set_weights(converted_w)
# f.close()
weights_path = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
model.load_weights(weights_path)
print('Model loaded.')

# # build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
# #
# # # note that it is necessary to start with a fully-trained
# # # classifier, including the top classifier,
# # # in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# # add the model on top of the convolutional base
model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:14]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1.,
    featurewise_center=True,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)

test_datagen = ImageDataGenerator(rescale=1., featurewise_center=True)
test_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('./models/forth_try.h5')