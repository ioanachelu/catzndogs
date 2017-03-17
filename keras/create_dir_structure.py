import numpy as np
import os
import tensorflow as tf
from shutil import copyfile


FLAGS = tf.app.flags.FLAGS

filenames = os.listdir("../train")
train_images = [os.path.join("../train", image) for image in filenames]

cat_images = []
dog_images = []
for image, filename in zip(train_images, filenames):
    if 'cat' not in image:
        dog_images.append([image, filename])
    else:
        cat_images.append([image, filename])

nb_images = len(cat_images)
nb_images_train_cat = int(nb_images * 8 / 10)

nb_images = len(dog_images)
nb_images_train_dog = int(nb_images * 8 / 10)

train_cats = cat_images[:nb_images_train_cat]
val_cats = cat_images[nb_images_train_cat:]

train_dogs = dog_images[:nb_images_train_dog]
val_dogs = dog_images[nb_images_train_dog:]

directories = ['./data/train/cats', './data/train/dogs', './data/validation/cats', './data/validation/dogs']
lists = [train_cats, train_dogs, val_cats, val_dogs]
for dir in directories:
    if not tf.gfile.Exists(dir):
        tf.gfile.MakeDirs(dir)
    else:
        tf.gfile.DeleteRecursively(dir)
        tf.gfile.MakeDirs(dir)
for dir, l in zip(directories, lists):
    for image, filename in l:
        new_image_path = os.path.join(dir, filename)
        copyfile(image, new_image_path)





