import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# hyper parameters for model
based_model_last_block_layer_number = 142  # value is based on based model selected.
img_width, img_height = 224, 224  # change based on the shape/structure of your images
batch_size = 128  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
top_model_weights_path = './models/top_model_weights_resnet_v2.h5'
final_model_weights_path = './models/model_weights_resnet_v2.h5'
model_path = './models/model_resnet_v2.json'
nb_train_samples = 20000
nb_validation_samples = 5000

def preprocess_input_resnet(x):
    from keras.applications.resnet50 import preprocess_input
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

# Pre-Trained CNN Model using imagenet dataset for pre-trained weights
base_model = ResNet50(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

# Top Model Block
x = base_model.output
# x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
predictions = Dense(1, activation='sigmoid')(x)

# add your top layer block to your base model
model = Model(base_model.input, predictions)
print(model.summary())

for layer in base_model.layers:
    layer.trainable = False

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=transformation_ratio,
                                   zoom_range=transformation_ratio,
                                   cval=transformation_ratio,
                                   horizontal_flip=True,
                                   vertical_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# os.makedirs(os.path.join(os.path.abspath(train_data_dir), '../preview'), exist_ok=True)
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')
# save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
# save_prefix='aug',
# save_format='jpeg')
# use the above 3 commented lines if you want to save and look at how the data augmentations look like

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(img_width, img_height),
                                                              batch_size=batch_size,
                                                              class_mode='binary')

model.compile(optimizer='nadam',
              loss='binary_crossentropy',  # categorical_crossentropy if multi-class classifier
              metrics=['accuracy'])


callbacks_list = [
    ModelCheckpoint(top_model_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_acc', patience=5, verbose=0)
]

# Train Simple CNN
model.fit_generator(train_generator,
                    samples_per_epoch=nb_train_samples // nb_epoch,
                    nb_epoch=nb_epoch / 5,
                    validation_data=validation_generator,
                    nb_val_samples=nb_validation_samples // nb_epoch,
                    callbacks=callbacks_list)

# verbose
print("\nStarting to Fine Tune Model\n")

# add the best weights from the train top model
# at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
# we re-load model weights to ensure the best epoch is selected and not the last one.
model.load_weights(top_model_weights_path)

# based_model_last_block_layer_number points to the layer in your model you want to train.
# For example if you want to train the last block of a 19 layer VGG16 model this should be 15
# If you want to train the last Two blocks of an Inception model it should be 172
# layers before this number will used the pre-trained weights, layers above and including this number
# will be re-trained based on the new data.
for layer in model.layers[:based_model_last_block_layer_number]:
    layer.trainable = False
for layer in model.layers[based_model_last_block_layer_number:]:
    layer.trainable = True

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(optimizer='nadam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# save weights of best training epoch: monitor either val_loss or val_acc

callbacks_list = [
    ModelCheckpoint(final_model_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=5, verbose=0)
]

# fine-tune the model
model.fit_generator(train_generator,
                    samples_per_epoch=nb_train_samples // nb_epoch,
                    nb_epoch=nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=nb_validation_samples // nb_epoch,
                    callbacks=callbacks_list)

# save model
model_json = model.to_json()
with open(model_path, 'w') as json_file:
    json_file.write(model_json)