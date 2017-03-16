import tensorflow as tf
import flags
import os
import numpy as np

FLAGS = tf.app.flags.FLAGS
from PIL import Image, ImageOps
import multiprocessing
import concurrent.futures
import sys


def preprocess():
    train_images = os.listdir(FLAGS.train_dir)
    train_images = [os.path.join("./train", image) for image in train_images]
    train_labels = []
    for image in train_images:
        if 'cat' not in image:
            train_labels.append(1)
        else:
            train_labels.append(0)

    dataset_train = list(zip(train_images, train_labels))
    np.random.shuffle(dataset_train)
    total_nr_training_examples = len(dataset_train)
    train_examples = int(total_nr_training_examples * 80 / 100)

    dataset_train, dataset_val = dataset_train[:train_examples], dataset_train[train_examples:]

    test_images = os.listdir(FLAGS.test_dir)
    test_images = [os.path.join("./test", image) for image in test_images]

    dataset_test = zip(test_images)

    print("started saving the preprocessed array")
    np.save("./train.npy", dataset_train)
    np.save("./val.npy", dataset_val)
    np.save("./test.npy", dataset_test)
    print("end saving the preprocessed array")
    return

preprocess()