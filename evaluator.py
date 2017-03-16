from multiprocessing import Process, Queue, Value
import flags
import tensorflow as tf
import time
import datetime
import numpy as np
import threading
import multiprocessing
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import flags
import os
from model import CatznDogs
from preprocessing import preprocess
from image_reader import ImageReader
from collections import deque
import os

FLAGS = tf.app.flags.FLAGS


class Evaluator():
    def __init__(self):
        self.stop = False

        if not tf.gfile.Exists(FLAGS.test_load_queue_path):
            self.settings = {"best_acc": None,
                             "best_checkpoint": None,
                             "last_checkpoint": None,
                             "acc_increasing": None,
                             "last_accs": deque()
                             }
        else:
            self.settings = np.load(FLAGS.test_load_queue_path)[()]

        self.setup()

    def setup(self):
        self.recreate_directory_structure()
        self.coord = tf.train.Coordinator()
        # Load reader.
        with tf.name_scope("create_inputs"):
            self.reader = ImageReader(
                "./val.npy",
                True,
                self.coord)
            self.image_batch, self.label_list_batch = self.reader.dequeue(FLAGS.batch_size)

        global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
        self.net = CatznDogs({'data': self.image_batch}, global_step)

        # Set up tf session and initialize variables.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Start queue threads.
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

        self.ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    def run(self):

        while not self.stop:

            all_models_paths = self.ckpt.all_model_checkpoint_paths
            index_current_model = list(all_models_paths).index(self.ckpt.model_checkpoint_path)
            if self.settings["last_checkpoint"]:
                index_last_evaluated_model = list(all_models_paths).index(self.settings["last_checkpoint"])
            else:
                index_last_evaluated_model = -1

            if index_current_model != index_last_evaluated_model:
                index_model_under_evaluation = index_last_evaluated_model + 1
                self.settings["last_checkpoint"] = all_models_paths[index_model_under_evaluation]

                print("Evaluator started evaluating")
                acc_pred = self.net.test(self.image_batch, self.label_list_batch, self.coord, self.sess,
                                         self.reader.nb_samples, checkpoint_iteration=index_model_under_evaluation)
                self.settings["last_accs"].append(acc_pred)

                if not self.settings["best_acc"] or acc_pred < self.settings["best_acc"]:
                    self.settings["best_acc"] = acc_pred
                    self.settings["best_checkpoint"] = self.settings["last_checkpoint"]
                    self.settings["acc_increasing"] = 0
                else:
                    self.settings["acc_increasing"] += 1

                if self.settings["acc_increasing"] >= 5:
                    self.stop = 1

                np.save(FLAGS.test_load_queue_path, self.settings)
                print("Best model is {} with best Acc {}".format(self.settings["best_checkpoint"],
                                                                 self.settings["best_acc"]))
            else:
                time.sleep(10)

        self.coord.request_stop()
        self.coord.join(self.threads)
        print("Best model is {} with best Acc {}".format(self.settings["best_checkpoint"], self.settings["best_acc"]))

    def recreate_directory_structure(self):
        if not tf.gfile.Exists(FLAGS.test_summaries_dir):
            tf.gfile.MakeDirs(FLAGS.test_summaries_dir)
        else:
            tf.gfile.DeleteRecursively(FLAGS.test_summaries_dir)
            tf.gfile.MakeDirs(FLAGS.test_summaries_dir)


def run():
    evaluator = Evaluator()
    evaluator.run()


if __name__ == '__main__':
    run()
