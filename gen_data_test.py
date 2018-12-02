import tensorflow as tf
import scipy.misc
import random
import numpy as np
from config import height, width
import cv2
import utils

def convert_to_coordinate(pts, width=width, height=height):
    return tuple( ((pts + 1) / 2 * [width, height]).astype(np.int32) )
logger = utils.get_logger()

def main():
    filename_queue = tf.train.string_input_producer(['../0.tfrecords'], num_epochs=None, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'stable': tf.VarLenFeature(tf.float32),
                                           'unstable': tf.VarLenFeature(tf.float32),
                                           'flow': tf.VarLenFeature(tf.float32),
                                           'feature_matches1': tf.VarLenFeature(tf.float32),
                                           'feature_matches2': tf.VarLenFeature(tf.float32),
                                       })
    stable_ = tf.reshape(tf.sparse_tensor_to_dense(features['stable']), [2, height, width, -1])
    unstable_ = tf.reshape(tf.sparse_tensor_to_dense(features['unstable']), [2, height, width, -1])
    flow_ = tf.reshape(tf.sparse_tensor_to_dense(features['flow']), [height, width, -1])[:, :, :2]

    feature_matches1_ = tf.reshape(tf.sparse_tensor_to_dense(features['feature_matches1']), [-1, 4])
    feature_matches2_ = tf.reshape(tf.sparse_tensor_to_dense(features['feature_matches2']), [-1, 4])

    sv = tf.train.Supervisor(logdir='./tmp/log', save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        stable, unstable, feature_matches1, feature_matches2 = sess.run([stable_, unstable_, feature_matches1_, feature_matches2_])
        feature_matches = [feature_matches1, feature_matches2]
        for i in range(2):
            si = (stable[i] + .5) * 255
            ui = (unstable[i] + .5) * 255
            ui = np.concatenate((si[..., -1, None], ui), axis=1)
            ui = np.concatenate([ui, ui, ui], axis=2)
            print(ui.shape)
            for match in feature_matches[i]:
                if np.random.uniform(0, 1) < 0.8: continue
                cv2.line(ui, \
                convert_to_coordinate(match[:2]), convert_to_coordinate(match[2:] + [2, 0]), tuple(np.random.rand(3) * 255))
            # si = np.transpose(si, [0, 2, 1]).reshape(height, width * si.shape[-1])
            for j in range(si.shape[-1]):
                cv2.imwrite(str(i)+str(j)+'.jpg', si[..., j])
            cv2.imwrite(str(i)+'-match.jpg', ui)


if '__main__' == __name__:
    main()