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

train_cats = cat_images[:1000]
val_cats = cat_images[1000:]
train_dogs = dog_images[:1000]
val_dogs = dog_images[1000:]

directories = ['./data/train/cats', './data/train/dogs', './data/val/cats', './data/val/dogs']
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





