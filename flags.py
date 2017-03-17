import tensorflow as tf

# Basic model parameters.
tf.app.flags.DEFINE_integer('checkpoint_every', 500,
                            """Checkpoint interval""")
tf.app.flags.DEFINE_integer('summary_every', 1000,
                            """Summary interval""")
tf.app.flags.DEFINE_integer('test_summary_every', 1,
                            """Summary interval""")
tf.app.flags.DEFINE_string('GPU', "0",
                           """The GPU device to run on""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Resume training from latest checkpoint""")
tf.app.flags.DEFINE_boolean('train', True,
                            """Whether to train or test""")
tf.app.flags.DEFINE_string('checkpoint_dir', './models/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_steps', 100000,
                            """Num of steps to train the network""")
tf.app.flags.DEFINE_string('summaries_dir', './summaries/',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('test_summaries_dir', './test_summaries/',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('train_dir', './train',
                           """Directory where the train dataset is""")
tf.app.flags.DEFINE_string('test_dir', './train',
                           """Directory where the test dataset is""")
tf.app.flags.DEFINE_string('pretrained_weights', './vgg16.npy',
                           """Path to where the pretrained  weights for VGG 16 reside""")
tf.app.flags.DEFINE_float('lr', 1e-4, """Learning rate""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9, """Moving average decay for loss""")
tf.app.flags.DEFINE_integer('stepsize', 70000, """Learning rate num steps per decay""")
tf.app.flags.DEFINE_float('gamma', 0.1, """Learning rate power for decay""")
tf.app.flags.DEFINE_float('momentum', 0.9, """Learning rate momentum""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005, """Weight decay for weights not biases""")
tf.app.flags.DEFINE_string('test_load_queue_path', "test_queue.npy", """Path to test queue""")
tf.flags.DEFINE_integer("resize_height", 150, """resize_height""")
tf.flags.DEFINE_integer("resize_width", 150, """resize_width""")
tf.flags.DEFINE_integer("height", 150, """height""")
tf.flags.DEFINE_integer("width", 150, """width""")
tf.app.flags.DEFINE_boolean('layer_norm', True, """Whether to use layer normalization or not""")
tf.app.flags.DEFINE_string('freeze_layers', "conv1_1 conv1_2 conv2_1 conv2_2 conv3_1 conv3_2 conv3_3 conv4_1 conv4_2 conv4_3",
                           """The layer names to freeze separated by spaces""")
