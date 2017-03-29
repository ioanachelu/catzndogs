
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers

seed = 42
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)


based_model_last_block_layer_number = 15
img_width, img_height = 224, 224
batch_size = 32
nb_epoch = 50
transformation_ratio = .05
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
top_model_weights_path = './models/top_model_weights_vgg.h5'
final_model_weights_path = './models/model_weights_vgg.h5'
model_path = './models/model_vgg.json'
nb_train_samples = 20000
nb_validation_samples = 5000

base_model = VGG16(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
model = Model(input=base_model.input, output=top_model(base_model.output))
model_json = model.to_json()
with open(model_path, 'w') as json_file:
    json_file.write(model_json)
print(model.summary())

for layer in base_model.layers:
    layer.trainable = False

train_datagen = ImageDataGenerator(rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(img_width, img_height),
                                                              batch_size=batch_size,
                                                              class_mode='binary')

model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks_list = [
    ModelCheckpoint(top_model_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_acc', patience=5, verbose=0)
]

# Train Simple CNN
model.fit_generator(train_generator,
                    samples_per_epoch=nb_train_samples,
                    nb_epoch=nb_epoch / 5,
                    validation_data=validation_generator,
                    nb_val_samples=nb_validation_samples,
                    callbacks=callbacks_list)

print("\nStarting to Fine Tune Model\n")
model.load_weights(top_model_weights_path)

for layer in model.layers[:based_model_last_block_layer_number]:
    layer.trainable = False
for layer in model.layers[based_model_last_block_layer_number:]:
    layer.trainable = True

opt = optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks_list = [
    ModelCheckpoint(final_model_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=5, verbose=0)
]

model.fit_generator(train_generator,
                    samples_per_epoch=nb_train_samples,
                    nb_epoch=nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=nb_validation_samples,
                    callbacks=callbacks_list)

model_json = model.to_json()
with open(model_path, 'w') as json_file:
    json_file.write(model_json)