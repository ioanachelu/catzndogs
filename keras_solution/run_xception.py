from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.xception import preprocess_input

seed = 42
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

based_model_last_block_layer_number = 126
img_width, img_height = 229, 229
batch_size = 32
nb_epoch = 50
transformation_ratio = .5
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
top_model_weights_path = './models/top_model_weights_inception_4.h5'
final_model_weights_path = './models/model_weights_inception_4.h5'
model_path = './models/model_inception_4.json'
nb_train_samples = 20000
nb_validation_samples = 5000


base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(base_model.input, predictions)
print(model.summary())

for layer in base_model.layers:
    layer.trainable = False

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')


validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(img_width, img_height),
                                                              batch_size=batch_size,
                                                              class_mode='binary')

model.compile(optimizer='nadam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks_list = [
    ModelCheckpoint(top_model_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_acc', patience=5, verbose=0)
]

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

opt = SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
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