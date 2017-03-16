import threading
import multiprocessing
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import flags
import os

from preprocessing import preprocess
from model import CatznDogs
from image_reader import ImageReader
import os
FLAGS = tf.app.flags.FLAGS


def recreate_directory_structure():
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
            tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    if not tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.MakeDirs(FLAGS.summaries_dir)
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
            tf.gfile.MakeDirs(FLAGS.summaries_dir)

    # if FLAGS.data_augmentation:
    if not tf.gfile.Exists("./train.npy") or \
            not tf.gfile.Exists("./test.npy"):
        preprocess()
    # else:
    #     if not tf.gfile.Exists(os.path.join(FLAGS.dataset_path, "train_no_flip.npy")) or \
    #             not tf.gfile.Exists(os.path.join(FLAGS.dataset_path, "test.npy")):
    #         preprocess_concurrent()

def run():
    recreate_directory_structure()
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            "./train.npy",
            True,
            coord)

        image_batch, label_list_batch = reader.dequeue(FLAGS.batch_size)

    global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
    net = CatznDogs({'data': image_batch}, global_step)
    net.train(image_batch, label_list_batch, coord)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    run()
