import tensorflow as tf
import scipy.misc
import random
from config import *

def warp_img(image, seed):
    image = tf.image.random_brightness(image, max_delta=32./255., seed = seed)
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed = seed)
    #image = tf.image.random_hue(image, max_delta=0.2, seed = seed)
    
    h = int(height / random_crop_rate)
    w = int(width / random_crop_rate)
    image = tf.image.resize_images(image, (h, w), method=seed % 4)
    image = tf.random_crop(image, (height, width, 1), seed = seed)
    
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed = seed)
    image = tf.image.random_flip_left_right(image, seed = seed)
    noise = np.random.normal(0,0.05,image.shape)
    #image = image + noise

    return tf.clip_by_value(image, -0.5, 0.5)

def read_and_decode(filename, num_epochs):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'x': tf.FixedLenFeature([height * width * tot_ch], tf.float32),
                                           'y': tf.FixedLenFeature([height * width], tf.float32)
                                       })
    x_ = tf.reshape(features['x'], [height, width, tot_ch])
    y_ = tf.reshape(features['y'], [height, width, 1])
    seed = random.randint(0, 2**31 - 1)
    y = warp_img(y_, seed)
    for i in range(tot_ch):
        temp = tf.slice(x_, [0, 0, i], [-1, -1, 1])
        if (i == 0):
            x = warp_img(temp, seed)
        else:
            x = tf.concat([x, warp_img(temp, seed)], 2)
    
    return x, y

def run():
    x, y = read_and_decode("data/train.tfrecords", 3)

    x_batch, y_batch = tf.train.shuffle_batch([x, y],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    init = tf.initialize_all_variables()
    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.initialize_local_variables())
        threads = tf.train.start_queue_runners(sess=sess, coord = coord)
        x_b, y_b = sess.run([x_batch, y_batch])
        print(x_b.shape)
        print(x_b)
        mage_summary = tf.summary.image('y', y_b, 5)
        for i in range(tot_ch):
            temp = tf.slice(x_b, [0, 0, 0, i], [-1, -1, -1, 1])
            mage_summary = tf.summary.image('x' + str(i), temp, 5)
        
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./log/', sess.graph)
        summary_all = sess.run(merged)
        summary_writer.add_summary(summary_all, 0)
        summary_writer.close()
