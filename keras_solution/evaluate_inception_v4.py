from keras.layers import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
import csv
from keras.models import model_from_json
from keras.applications.xception import preprocess_input
seed = 42
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

based_model_last_block_layer_number = 400
img_width, img_height = 229, 229
batch_size = 1
nb_epoch = 50
learn_rate = 1e-4
momentum = .9
transformation_ratio = .2
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
final_model_weights_path = './models/model_weights_inception_v4.h5'
model_path = './models/model_inception_v4.json'
nb_train_samples = 20000
nb_validation_samples = 5000
test_data_dir = './data/test'
results_name = 'submission_inception_v4.csv'

json_file = open(model_path, 'r')

loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(final_model_weights_path)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  shuffle=False)

y_probabilities = model.predict_generator(test_generator,
                                          val_samples=12500)
y_probabilities = [p[0] for p in y_probabilities]


filenames = [filename.split('/')[1] for filename in test_generator.filenames]
ids = [int(filename.split('.')[0]) for filename in filenames]

submission = list(zip(ids, y_probabilities))
submission.sort(key=lambda t: t[0])

with open(results_name, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(('id', 'label'))
    writer.writerows(submission)
