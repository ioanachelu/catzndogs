import tensorflow as tf
import numpy as np
import flags
FLAGS = tf.app.flags.FLAGS

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

def read_labeled_image_list_from_npy(data_npy):
    """Reads txt file containing paths to images and ground truth label lists.

    Args:
      data_dir: path to the directory with images and gt lists.
      data_list: path to the file with lines of the form '/path/to/image list_of_gts'.

    Returns:
      Two lists with all file names for images and label lists, respectively.
    """
    dataset = np.load(data_npy)
    images = []
    labels = []
    print("Dataset consists of {} images and gts".format(len(dataset)))
    for image, label in dataset:
        images.append(image)
        labels.append(int(label))
    return images, labels


def read_images_from_disk(input_queue, train):
    """Read one image and its corresponding gts with pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its gts.

    Returns:
      Two tensors: the decoded image and its gts.
    """

    img_contents = tf.read_file(input_queue[0])
    label = input_queue[1]

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img = tf.cast(img, dtype=tf.float32)

    # Resize image.
    if FLAGS.resize_height:
        img = tf.image.resize_images(img,
                                       size=[FLAGS.resize_height, FLAGS.resize_width],
                                       method=tf.image.ResizeMethod.BILINEAR)

    # Crop to final dimensions.
    if train:
        img = tf.random_crop(img, [FLAGS.height, FLAGS.width, 3])
    else:
        # Central crop, assuming resize_height > height, resize_width > width.
        img = tf.image.resize_image_with_crop_or_pad(img, FLAGS.height, FLAGS.width)

    # # Randomly distort the image.
    # if train:
    #     image = distort_image(image, thread_id)

    # Extract mean.
    img -= IMG_MEAN

    return img, label


class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding gt obstacles
       from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_npy, shuffle, coord):
        '''Initialise an ImageReader.

        Args:
          data_list: path to the file with lines of the form '/path/to/image list_of_gts'.
          coord: TensorFlow queue coordinator.
        '''
        self.data_npy = data_npy
        self.coord = coord
        self.shuffle = shuffle

        self.image_list, self.label_list = read_labeled_image_list_from_npy(self.data_npy)
        self.nb_samples = len(self.image_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.int32)
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=shuffle)  # not shuffling if it is val
        self.image, self.label = read_images_from_disk(self.queue, train=shuffle)

    def replenish_queue(self):
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=False)
        self.image, self.label = read_images_from_disk(self.queue)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch, label_batch = tf.train.batch([self.image, self.label],
                                                  num_elements)
        return image_batch, label_batch
