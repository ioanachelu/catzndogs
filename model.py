import tensorflow as tf
import flags
from image_reader import ImageReader
import os
from network import Network
import time
import numpy as np
from PIL import Image, ImageDraw

IMG_MEAN = np.array((98, 97, 101), dtype=np.float32)

FLAGS = tf.app.flags.FLAGS


class CatznDogs(Network):
    def __init__(self, inputs, global_step, model_type="vgg"):
        self.layer_norm = FLAGS.layer_norm
        super(CatznDogs, self).__init__(inputs, global_step, model_type=model_type)
        if FLAGS.freeze_layers:
            self.freezed_layers = FLAGS.freeze_layers.split(" ")

    def setup(self, model_type="vgg"):

        if model_type == "vgg":
            (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', layer_norm=self.layer_norm)
             .conv(3, 3, 64, 1, 1, name='conv1_2', layer_norm=self.layer_norm)
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', layer_norm=self.layer_norm)
             .conv(3, 3, 128, 1, 1, name='conv2_2', layer_norm=self.layer_norm)
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1', layer_norm=self.layer_norm)
             .conv(3, 3, 256, 1, 1, name='conv3_2', layer_norm=self.layer_norm)
             .conv(3, 3, 256, 1, 1, name='conv3_3', layer_norm=self.layer_norm)
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1', layer_norm=self.layer_norm)
             .conv(3, 3, 512, 1, 1, name='conv4_2', layer_norm=self.layer_norm)
             .conv(3, 3, 512, 1, 1, name='conv4_3', layer_norm=self.layer_norm)
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1', layer_norm=self.layer_norm)
             .conv(3, 3, 512, 1, 1, name='conv5_2', layer_norm=self.layer_norm)
             .conv(3, 3, 512, 1, 1, name='conv5_3', layer_norm=self.layer_norm)
             .max_pool(2, 2, 2, 2, name='pool5')
             .fc(4096, name="fc6", layer_norm=self.layer_norm)
             .dropout(0.5, name="drop6")
             .fc(4096, name="fc7", layer_norm=self.layer_norm)
             .dropout(0.5, name="drop7")
             .fc(2, name="cls_score", weight_initializer=tf.contrib.layers.xavier_initializer(),
                 bias_initializer=tf.zeros_initializer(), layer_norm=self.layer_norm)
             )
        else:
            (self.feed('data')
             .conv(7, 7, 64, 2, 2, biased=False, name='conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(1, 1, 256, 1, 1, biased=False, name='res2a_branch1'))

            (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, name='res2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, name='res2a_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c'))

            (self.feed('res2a_branch1',
                       'res2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, biased=False, name='res2b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, name='res2b_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c'))

            (self.feed('res2a_relu',
                       'res2b_branch2c')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 64, 1, 1, biased=False, name='res2c_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, name='res2c_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c'))

            (self.feed('res2b_relu',
                       'res2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1'))

            (self.feed('res2c_relu')
             .conv(1, 1, 128, 2, 2, biased=False, name='res3a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, name='res3a_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c'))

            (self.feed('res3a_branch1',
                       'res3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(1, 1, 128, 1, 1, biased=False, name='res3b1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, name='res3b1_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c'))

            (self.feed('res3a_relu',
                       'res3b1_branch2c')
             .add(name='res3b1')
             .relu(name='res3b1_relu')
             .conv(1, 1, 128, 1, 1, biased=False, name='res3b2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, name='res3b2_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c'))

            (self.feed('res3b1_relu',
                       'res3b2_branch2c')
             .add(name='res3b2')
             .relu(name='res3b2_relu')
             .conv(1, 1, 128, 1, 1, biased=False, name='res3b3_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, name='res3b3_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c'))

            (self.feed('res3b2_relu',
                       'res3b3_branch2c')
             .add(name='res3b3')
             .relu(name='res3b3_relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1'))

            (self.feed('res3b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4a_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4a_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c'))

            (self.feed('res4a_branch1',
                       'res4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b1_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b1_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c'))

            (self.feed('res4a_relu',
                       'res4b1_branch2c')
             .add(name='res4b1')
             .relu(name='res4b1_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b2_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b2_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c'))

            (self.feed('res4b1_relu',
                       'res4b2_branch2c')
             .add(name='res4b2')
             .relu(name='res4b2_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b3_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b3_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c'))

            (self.feed('res4b2_relu',
                       'res4b3_branch2c')
             .add(name='res4b3')
             .relu(name='res4b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b4_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b4_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c'))

            (self.feed('res4b3_relu',
                       'res4b4_branch2c')
             .add(name='res4b4')
             .relu(name='res4b4_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b5_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b5_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c'))

            (self.feed('res4b4_relu',
                       'res4b5_branch2c')
             .add(name='res4b5')
             .relu(name='res4b5_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b6_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b6_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c'))

            (self.feed('res4b5_relu',
                       'res4b6_branch2c')
             .add(name='res4b6')
             .relu(name='res4b6_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b7_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b7_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c'))

            (self.feed('res4b6_relu',
                       'res4b7_branch2c')
             .add(name='res4b7')
             .relu(name='res4b7_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b8_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b8_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c'))

            (self.feed('res4b7_relu',
                       'res4b8_branch2c')
             .add(name='res4b8')
             .relu(name='res4b8_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b9_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b9_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c'))

            (self.feed('res4b8_relu',
                       'res4b9_branch2c')
             .add(name='res4b9')
             .relu(name='res4b9_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b10_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b10_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c'))

            (self.feed('res4b9_relu',
                       'res4b10_branch2c')
             .add(name='res4b10')
             .relu(name='res4b10_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b11_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b11_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c'))

            (self.feed('res4b10_relu',
                       'res4b11_branch2c')
             .add(name='res4b11')
             .relu(name='res4b11_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b12_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b12_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c'))

            (self.feed('res4b11_relu',
                       'res4b12_branch2c')
             .add(name='res4b12')
             .relu(name='res4b12_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b13_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b13_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c'))

            (self.feed('res4b12_relu',
                       'res4b13_branch2c')
             .add(name='res4b13')
             .relu(name='res4b13_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b14_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b14_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c'))

            (self.feed('res4b13_relu',
                       'res4b14_branch2c')
             .add(name='res4b14')
             .relu(name='res4b14_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b15_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b15_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c'))

            (self.feed('res4b14_relu',
                       'res4b15_branch2c')
             .add(name='res4b15')
             .relu(name='res4b15_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b16_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b16_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c'))

            (self.feed('res4b15_relu',
                       'res4b16_branch2c')
             .add(name='res4b16')
             .relu(name='res4b16_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b17_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b17_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c'))

            (self.feed('res4b16_relu',
                       'res4b17_branch2c')
             .add(name='res4b17')
             .relu(name='res4b17_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b18_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b18_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c'))

            (self.feed('res4b17_relu',
                       'res4b18_branch2c')
             .add(name='res4b18')
             .relu(name='res4b18_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b19_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b19_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c'))

            (self.feed('res4b18_relu',
                       'res4b19_branch2c')
             .add(name='res4b19')
             .relu(name='res4b19_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b20_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b20_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c'))

            (self.feed('res4b19_relu',
                       'res4b20_branch2c')
             .add(name='res4b20')
             .relu(name='res4b20_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b21_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b21_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c'))

            (self.feed('res4b20_relu',
                       'res4b21_branch2c')
             .add(name='res4b21')
             .relu(name='res4b21_relu')
             .conv(1, 1, 256, 1, 1, biased=False, name='res4b22_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, name='res4b22_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c'))

            (self.feed('res4b21_relu',
                       'res4b22_branch2c')
             .add(name='res4b22')
             .relu(name='res4b22_relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1'))

            (self.feed('res4b22_relu')
             .conv(1, 1, 512, 1, 1, biased=False, name='res5a_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, name='res5a_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c'))

            (self.feed('res5a_branch1',
                       'res5a_branch2c')
             .add(name='res5a')
             .relu(name='res5a_relu')
             .conv(1, 1, 512, 1, 1, biased=False, name='res5b_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, name='res5b_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c'))

            (self.feed('res5a_relu',
                       'res5b_branch2c')
             .add(name='res5b')
             .relu(name='res5b_relu')
             .conv(1, 1, 512, 1, 1, biased=False, name='res5c_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, name='res5c_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c'))

            (self.feed('res5b_relu',
                       'res5c_branch2c')
             .add(name='res5c')
             .relu(name='res5c_relu')
             .atrous_conv(3, 3, 21, 6, padding='SAME', relu=False, name='fc1_voc12_c0'))

            (self.feed('res5c_relu')
             .atrous_conv(3, 3, 21, 12, padding='SAME', relu=False, name='fc1_voc12_c1'))

            (self.feed('res5c_relu')
             .atrous_conv(3, 3, 21, 18, padding='SAME', relu=False, name='fc1_voc12_c2'))

            (self.feed('res5c_relu')
             .atrous_conv(3, 3, 21, 24, padding='SAME', relu=False, name='fc1_voc12_c3'))

            (self.feed('fc1_voc12_c0',
                       'fc1_voc12_c1',
                       'fc1_voc12_c2',
                       'fc1_voc12_c3')
             .add(name='fc1_voc12'))

    def train(self, image_batch, label_batch, coord):
        # Predictions.
        logits = self.layers['cls_score']
        preds_batch = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [-1])
        accuracy, update_op = self.accuracy(preds_batch, label_batch)
        step_accuracy, _ = self.step_accuracy(preds_batch, label_batch)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_batch)
        l2_losses = [FLAGS.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
        reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

        images_summary = tf.py_func(self.inv_preprocess, [image_batch, 1], tf.uint8)

        total_image_summary = tf.summary.image('images', images_summary, max_outputs=4)  # Concatenate row-wise.
        total_summary = []
        total_summary.append(tf.summary.scalar('Loss_raw', reduced_loss))
        total_summary.append(tf.summary.scalar('Accuracy', accuracy))

        activations = tf.GraphKeys.ACTIVATIONS
        total_summary.append(tf.contrib.layers.summarize_collection(tf.GraphKeys.ACTIVATIONS,
                                                                    summarizer=tf.contrib.layers.summarize_activation))
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, name='loss_avg')
        maintain_averages_op = ema.apply([reduced_loss, step_accuracy])
        total_summary.append(tf.summary.scalar('Avg_loss', ema.average(reduced_loss)))
        step_avg_accuracy = ema.average(step_accuracy)
        summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir, graph=tf.get_default_graph())

        # Define loss and optimisation parameters.
        learning_rate = tf.train.exponential_decay(FLAGS.lr, self.global_step,
                                                   FLAGS.stepsize, FLAGS.gamma, staircase=True)
        total_summary.append(tf.summary.scalar('LR', learning_rate))

        all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]

        if FLAGS.freeze_layers:
            for freezed in self.freezed_layers:
                all_trainable = [v for v in all_trainable if freezed not in v.name]

        with tf.control_dependencies([maintain_averages_op]):
            # opt = tf.train.RMSPropOptimizer(learning_rate, FLAGS.momentum, 0.0, 1e-6)
            opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
            grads = tf.gradients(reduced_loss, all_trainable)

            for grad, weight in zip(grads, all_trainable):
                total_summary.append(tf.summary.histogram(weight.name + '_grad', grad))
                total_summary.append(tf.summary.histogram(weight.name, weight))

            train_op = opt.apply_gradients(zip(grads, all_trainable))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        merged_summary = tf.summary.merge(total_summary)

        increment_global_step = self.global_step.assign_add(1)

        restore_var = tf.global_variables()
        if FLAGS.resume:
            loader = tf.train.Saver(var_list=restore_var)
            self.load_model(loader, sess)
            sess.run(tf.local_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # Loading .npy weights.
            self.load(FLAGS.pretrained_weights, sess, ignore_missing=True)

        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=restore_var, max_to_keep=1000)
        # Iterate over training steps.
        step_count = sess.run(self.global_step)
        while True:
            if step_count == FLAGS.num_steps:
                break
            start_time = time.time()

            if step_count % FLAGS.checkpoint_every == 0:
                loss_value, acc_pred, step_acc_pred, step_avg_acc_pred, _, images, labels, preds, summary_images, summary, _, _ = sess.run(
                    [reduced_loss, accuracy, step_accuracy, step_avg_accuracy, update_op, image_batch, label_batch, preds_batch,
                     total_image_summary,
                     merged_summary, train_op, increment_global_step])
                summary_writer.add_summary(summary_images, step_count)
                summary_writer.add_summary(summary, step_count)
                self.save_model(saver, sess, FLAGS.checkpoint_dir, step_count, self.global_step)
            elif step_count % FLAGS.summary_every == 0:
                loss_value, acc_pred, step_acc_pred, step_avg_acc_pred, _, images, labels, preds, summary_images, summary, _, _ = sess.run(
                    [reduced_loss, accuracy, step_accuracy, step_avg_accuracy, update_op, image_batch, label_batch, preds_batch,
                     total_image_summary,
                     merged_summary, train_op, increment_global_step])
                summary_writer.add_summary(summary_images, step_count)
                summary_writer.add_summary(summary, step_count)
            else:
                loss_value, acc_pred, step_acc_pred, step_avg_acc_pred, _, _, _ = sess.run(
                    [reduced_loss, accuracy, step_accuracy, step_avg_accuracy, update_op, train_op, increment_global_step])
            duration = time.time() - start_time
            #if step_count % FLAGS.summary_every == 0:
            print('step {:d} \t loss = {:.3f}, step_acc = {:.3f}, acc_exp_avg = {:.3f}, running_avg_acc = {:.3f}, ({:.3f} sec/step)'.format(
                        step_count, loss_value, step_acc_pred, step_avg_acc_pred, acc_pred,
                        duration))
            print("labels vs pred {}".format(list(zip(labels, preds))))
            step_count += 1
        coord.request_stop()
        coord.join(threads)

    def test(self, image_batch, label_batch, coord, sess, nb_test_samples, checkpoint_iteration=None):
        # Which variables to load.v
        restore_var = tf.global_variables()

        # Predictions.
        logits = self.layers['cls_score']
        label_batch = tf.reshape(label_batch, [-1])
        acc, update_op = self.accuracy(logits, label_batch)

        images_summary = tf.py_func(self.inv_preprocess, [image_batch, 1], tf.uint8)
        image_pred_summary = tf.py_func(self.draw_pred_on_img, [image_batch, tf.argmax(logits, 1), label_batch, 1],
                                        tf.uint8)

        total_image_summary = tf.summary.image('images',
                                               tf.concat([images_summary, image_pred_summary], 2),
                                               max_outputs=1)  # Concatenate row-wise.

        summary_writer = tf.summary.FileWriter(FLAGS.test_summaries_dir, graph=tf.get_default_graph())

        init = tf.global_variables_initializer()

        sess.run(init)
        sess.run(tf.local_variables_initializer())

        # Load weights.
        loader = tf.train.Saver(var_list=restore_var)
        self.load_model(loader, sess, checkpoint_iteration)

        # Iterate over testing steps.
        for step in range(nb_test_samples):
            if step % FLAGS.test_summary_every == 0:
                acc_pred, _, summary_images = sess.run([acc, update_op, total_image_summary])
                summary_writer.add_summary(summary_images, step)
                # print('Mean acc: {:.3f}'.format(acc_pred))
            else:
                acc_pred, _ = sess.run([acc, update_op])
        print('Mean accE: {:.3f}'.format(acc_pred))

        return acc_pred

    def load_model(self, saver, sess, checkpoint_iteration=None):
        '''Load trained weights.

        Args:
          saver: TensorFlow Saver object.
          sess: TensorFlow session.
          ckpt_path: path to checkpoint file with parameters.
        '''
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if checkpoint_iteration is not None:
            saver.restore(sess, ckpt.all_model_checkpoint_paths[checkpoint_iteration])
            print("Restored model parameters from {}".format(ckpt.all_model_checkpoint_paths[checkpoint_iteration]))
        else:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Restored model parameters from {}".format(ckpt.model_checkpoint_path))

    def save_model(self, saver, sess, logdir, step_evaled, step):
        '''Save weights.

        Args:
          saver: TensorFlow Saver object.
          sess: TensorFlow session.
          logdir: path to the snapshots directory.
          step: current training step.
        '''
        model_name = 'model-{}.ckpt'.format(step_evaled)
        checkpoint_path = os.path.join(logdir, model_name)

        if not os.path.exists(logdir):
            os.makedirs(logdir)
        saver.save(sess, checkpoint_path, global_step=step)
        print("Saved Model at {}".format(checkpoint_path))

    def inv_preprocess(self, imgs, num_images):
        """Inverse preprocessing of the batch of images.
           Add the mean vector and convert from BGR to RGB.

        Args:
          imgs: batch of input images.
          num_images: number of images to apply the inverse transformations on.

        Returns:
          The batch of the size num_images with the same spatial dimensions as the input.
        """
        n, h, w, c = imgs.shape
        assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
            n, num_images)
        outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
        for i in range(num_images):
            outputs[i] = (imgs[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8)
        return outputs

    def draw_pred_on_img(self, imgs, predictions, label_batch, num_images):
        """Inverse preprocessing of the batch of images.
           Add the mean vector and convert from BGR to RGB.
           Then draw predictions on the image

        Args:
          imgs: batch of input images.
          predictions: batch of predictions to draw on the imgs
          num_images: number of images to apply the inverse transformations on.

        Returns:
          The batch of the size num_images with the same spatial dimensions as the input.
        """
        nb_labels, = predictions.shape
        n, h, w, c = imgs.shape

        assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
            n, num_images)
        outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
        for i in range(num_images):
            outputs[i] = (imgs[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8)

        outputs_img = Image.fromarray(outputs[0])
        outputs_img_draw = ImageDraw.Draw(outputs_img)

        dr = outputs_img_draw
        for i, p in enumerate(predictions):
            dr.rectangle((24 + i, p, 24 + i + 1, p + 1), fill="red", outline="red")
            dr.rectangle((24 + i, label_batch[i], 24 + i + 1, label_batch[i] + 1), fill="blue", outline="blue")

        return np.expand_dims(outputs_img, 0)

    def accuracy(self, y_pred, y_true):
        return tf.metrics.accuracy(y_true, y_pred, name="accuracy")

    def step_accuracy(self, y_pred, y_true):
        return tf.metrics.accuracy(y_true, y_pred, name="step_accuracy")
