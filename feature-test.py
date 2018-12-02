from get_data_flow import *
import feature_fetcher
if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer(['./data3/test/0.tfrecords'], num_epochs=1, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'stable': tf.FixedLenFeature([height * width * (before_ch + 2)], tf.float32),
                                           'unstable': tf.FixedLenFeature([height * width * (after_ch + 2)], tf.float32),
                                           'flow': tf.FixedLenFeature([height * width * 2], tf.float32),
                                           'feature_matches1': tf.VarLenFeature(tf.float32),
                                           'feature_matches2': tf.VarLenFeature(tf.float32),
                                       })
    matches_tensor = tf.reshape(tf.sparse_tensor_to_dense(features['feature_matches1']), [-1, 4])
    matches_tensor2 = tf.reshape(tf.sparse_tensor_to_dense(features['feature_matches2']), [-1, 4])
    sv = tf.train.Supervisor(logdir='./tmp/log', save_summaries_secs=0, saver=None)
    with sv.managed_session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3))) as sess:
        matches1, matches2 = sess.run([matches_tensor, matches_tensor2])
        print(matches1, matches1.shape)
        gt = feature_fetcher.fetch('6.mp4.avi', 7)
        print(gt, gt.shape)
        print(matches2, matches2.shape)
        gt = feature_fetcher.fetch('6.mp4.avi', 8)
        print(gt, gt.shape)
