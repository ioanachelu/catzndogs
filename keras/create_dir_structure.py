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
tf.gfile.DeleteRecursively('./data/train/cats')
tf.gfile.MakeDirs('./data/train/cats')
tf.gfile.DeleteRecursively('./data/train/dogs')
tf.gfile.MakeDirs('./data/train/dogs')
tf.gfile.DeleteRecursively('./data/val/cats')
tf.gfile.MakeDirs('./data/val/cats')
tf.gfile.DeleteRecursively('./data/val/dogs')
tf.gfile.MakeDirs('./data/val/dogs')

for image, filename in train_cats:
    new_image_path = os.path.join("./data/train/cats", filename)
    copyfile(image, new_image_path)

for image, filename in train_dogs:
    new_image_path = os.path.join("./data/train/dogs", filename)
    copyfile(image, new_image_path)

for image, filename in val_cats:
    new_image_path = os.path.join("./data/val/cats", filename)
    copyfile(image, new_image_path)

for image, filename in val_dogs:
    new_image_path = os.path.join("./data/val/dogs", filename)
    copyfile(image, new_image_path)




