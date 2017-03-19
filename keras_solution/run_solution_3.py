from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
import numpy as np

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
model = applications.VGG16(input_shape=(150, 150, 3), weights='imagenet', include_top=False)
print('Model loaded.')

# # build a classifier model to put on top of the convolutional model
# top_model = Sequential()
# top_model.add(Flatten(input_shape=model.output_shape[1:]))
# top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(1, activation='sigmoid'))
# #
# # # note that it is necessary to start with a fully-trained
# # # classifier, including the top classifier,
# # # in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)

# top_model_weights_weights = top_model.get_weights()

# layer_dict = dict([(layer.name, layer) for layer in model.layers])
# weights = layer_dict['some_name'].get_weights()

# # add the model on top of the convolutional base
# model.add(top_model)

x = model.output
x = Flatten(input_shape=model.output_shape[1:])(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(1, activation='sigmoid')(x)

top_model = Model(inputs=model.input, outputs=preds)
top_model.load_weights(top_model_weights_path, by_name=True)


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in top_model.layers[:14]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
top_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
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
top_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples)

top_model.save_weights('./models/third_try.h5')