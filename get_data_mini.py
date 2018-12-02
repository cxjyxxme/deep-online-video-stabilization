import tensorflow as tf
import scipy.misc
import random
from config import *
import utils
logger = utils.get_logger()
assert(after_ch == 0)
def get_rand_para(seed): 
    h = int(height / random_crop_rate)
    w = int(width / random_crop_rate)
    hh = tf.random_uniform([], minval=0, maxval=h - height, dtype=tf.int32, seed=seed)
    ww = tf.random_uniform([], minval=0, maxval=w - width, dtype=tf.int32, seed=seed)
    return {"h": hh, "w": ww, "flip": (hh + ww) % 2}

def warp_img(image, seed, para):
    h = int(height / random_crop_rate)
    w = int(width / random_crop_rate)
    image = tf.image.resize_images(image, (h, w), method=tf.image.ResizeMethod.BILINEAR)
    image = tf.slice(image, [para['h'], para['w'], 0], [height, width, 1])
    
    image = tf.cond(tf.equal(para['flip'], 0), lambda: image, lambda: tf.image.flip_left_right(image))
    
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed = seed)
    image = tf.image.random_brightness(image, max_delta=32./255., seed = seed)
    ''' #random noise
    noise = np.random.normal(0,0.05,image.shape)
    image = image + noise
    '''

    return tf.clip_by_value(image, -0.5, 0.5)

def warp_flow(flow, para):
    flow_x = tf.slice(flow, [0, 0, 0], [-1, -1, 1])
    flow_y = tf.slice(flow, [0, 0, 1], [-1, -1, 1])
    h = int(height / random_crop_rate)
    w = int(width / random_crop_rate)
    flow_x = tf.image.resize_images(flow_x, (h, w), method=tf.image.ResizeMethod.BILINEAR)
    flow_y = tf.image.resize_images(flow_y, (h, w), method=tf.image.ResizeMethod.BILINEAR)
    flow_x = tf.slice(flow_x, [para['h'], para['w'], 0], [height, width, 1])
    flow_y = tf.slice(flow_y, [para['h'], para['w'], 0], [height, width, 1])
    flow_x = (flow_x + (1 - tf.cast(para['w'], tf.float32) / w * 2)) / (height / float(h)) - 1
    flow_y = (flow_y + (1 - tf.cast(para['h'], tf.float32) / h * 2)) / (width / float(w)) - 1

    fliped_y = tf.image.flip_left_right(flow_y)
    fliped_x = tf.image.flip_left_right(flow_x) * (-1) - 1.0 / width

    flow_x = tf.cond(tf.equal(para['flip'], 0), lambda: flow_x, lambda: fliped_x)
    flow_y = tf.cond(tf.equal(para['flip'], 0), lambda: flow_y, lambda: fliped_y)
    return tf.concat([flow_x, flow_y], axis=2)

def warp_point(points, mask, para):
    h = int(height / random_crop_rate)
    w = int(width / random_crop_rate)

    # points = points / [width, height, width, height] * 2 - 1
    points_x = tf.stack([points[:,0], points[:,2]], axis=1)
    points_y = tf.stack([points[:,1], points[:,3]], axis=1)
    points_x = (points_x + (1 - tf.cast(para['w'], tf.float32) / w * 2)) / (height / float(h)) - 1
    points_y = (points_y + (1 - tf.cast(para['h'], tf.float32) / h * 2)) / (width / float(w)) - 1

    fliped_x = points_x * (-1) - 1.0 / width
    points_x = tf.cond(tf.equal(para['flip'], 0), lambda: points_x, lambda: fliped_x)
    points = tf.stack([points_x[:,0], points_y[:,0], points_x[:,1], points_y[:,1]], axis=1)
    mask = tf.logical_and(tf.reduce_all(tf.logical_and(points >= -1, points <= 1), axis=1), mask)
    logger.info('points.shape, mask.shape={},{}'.format(points.shape, mask.shape))
    return points, mask

def get_rand_H(is_first, last_H):
    H = tf.random_uniform([1], minval=rand_H_min[0, 0], maxval=rand_H_max[0, 0], dtype=tf.float32)
    for i in range(3):
        for j in range(3):
            if (i == 0 and j == 0):
                continue
            H = tf.concat([H, tf.random_uniform([1], minval=rand_H_min[i, j], maxval=rand_H_max[i, j], dtype=tf.float32)], axis=0)
    if (is_first):
        return tf.reshape(H, [3, 3])
    else:
        return tf.reshape(H, [3, 3]) * rand_H_change_rate + last_H * (1 - rand_H_change_rate)

def mesh_grid(height, width):
    with tf.variable_scope('_meshgrid'):
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], 0)
    return grid

def get_rand_mask(is_first, last_H):
    H = get_rand_H(is_first, last_H)
    grid = mesh_grid(height, width)
    T_g = tf.matmul(H, grid)
    x_s = tf.slice(T_g, [0, 0], [1, -1])
    y_s = tf.slice(T_g, [1, 0], [1, -1])
    z_s = tf.slice(T_g, [2, 0], [1, -1])
    x_s_flat = tf.reshape(x_s / z_s, [-1])   
    y_s_flat = tf.reshape(y_s / z_s, [-1])

    t_1 = tf.ones(shape = tf.shape(x_s_flat))
    t_0 = tf.zeros(shape = tf.shape(x_s_flat))
    cond = tf.logical_or(tf.logical_or(tf.greater(t_1 * -1, x_s_flat), tf.greater(x_s_flat, t_1)), 
                         tf.logical_or(tf.greater(t_1 * -1, y_s_flat), tf.greater(y_s_flat, t_1)))
    black_pix = tf.reshape(tf.where(cond, t_1, t_0), [height, width])
    return black_pix, H

def get_rand_black_mask():
    max_dh = int(height * max_crop_rate / 2)
    max_dh = max(0, max_dh)
    max_dw = int(width * max_crop_rate / 2)
    max_dw = max(0, max_dw)

    h = tf.random_uniform([1], minval=0, maxval=max_dh + 1, dtype=tf.int32)
    w = tf.random_uniform([1], minval=0, maxval=max_dw + 1, dtype=tf.int32)
   
    zero = tf.zeros(shape=[1, 1])
    one =  tf.ones(shape=[1,1])
    mask = tf.tile(zero, tf.concat([h, tf.ones([1], dtype=tf.int32) * width], axis=0))
    temp = tf.concat([tf.tile(zero, tf.concat([height - h * 2, w], axis=0)), tf.tile(one, tf.concat([height - h * 2, width - w * 2], axis=0)), tf.tile(zero, tf.concat([height - h * 2, w], axis=0))], axis=1)
    mask = tf.concat([mask, temp, mask], axis = 0)
    mask = tf.reshape(mask, [height, width, 1])
    return mask 

def add_mask(pics):
    is_first = True
    last_H = tf.zeros([3, 3]) 
    for i in range(before_ch):
        temp = tf.reshape(tf.slice(pics, [0, 0, i], [-1, -1, 1]), [height, width])
        mask, last_H = get_rand_mask(is_first, last_H)
        is_first = False
        temp = temp * (1 -  mask) + mask * -1
        temp = tf.expand_dims(temp, 2)
        mask = tf.expand_dims(mask, 2)
        if (i == 0):
            ans = temp
            ans_mask = mask
        else:
            ans = tf.concat([ans, temp], axis = 2)
            ans_mask = tf.concat([ans_mask, mask], axis=2)

    if input_mask:
        return tf.concat([ans_mask, ans], axis=2)
    else:
        return ans

def get_img(path, pos):
    image = tf.image.decode_jpeg( tf.read_file(tf.string_join([path, tf.as_string(pos), '.jpg'])))
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, (height, width), 0)
    image = image - 0.5
    image = tf.reshape(image, [1, height, width, 1])
    return image

def read_and_decode(filepath, num_epochs, shuffle=True):
    file_obj = open(filepath + 'list.txt')
    file_txt = file_obj.read()
    file_list = []
    for f in file_txt.split(' '):
        file_list.append(filepath + f.strip())
    filename_queue = tf.train.string_input_producer(file_list, num_epochs=None, shuffle=shuffle)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'stable_path': tf.FixedLenFeature([], tf.string),
                                           'unstable_path': tf.FixedLenFeature([], tf.string),
                                           'pos': tf.FixedLenFeature([], tf.int64),
                                           'flow': tf.VarLenFeature(tf.float32),
                                           'feature_matches1': tf.VarLenFeature(tf.float32),
                                           'feature_matches2': tf.VarLenFeature(tf.float32),
                                       })
    pos = tf.cast(features['pos'], tf.int64)
    unstable_ = tf.concat([get_img(features['unstable_path'], pos - 1), get_img(features['unstable_path'], pos)], axis=0)
    stable1 = []
    stable2 = []
    for i in indices:
        stable1.append(get_img(features['stable_path'], pos - 1 - i))
        stable2.append(get_img(features['stable_path'], pos - i))
    stable1_ = tf.concat(stable1, axis=3)
    stable2_ = tf.concat(stable2, axis=3)
    stable_ = tf.concat([stable1_, stable2_], axis=0)

    #stable_input = tf.reshape(tf.sparse_tensor_to_dense(features['stable']), [2, height, width, -1])
    #stable_ = tf.concat((stable_input[..., 0, None], stable_input[:, :, :, -before_ch:]), axis=3)
    #unstable_ = tf.reshape(tf.sparse_tensor_to_dense(features['unstable']), [2, height, width, -1])#[:, :, :after_ch + 2]
    with tf.control_dependencies([
                                tf.assert_equal(tf.shape(stable_[0]), tf.constant((height, width, before_ch + 1))),
                                tf.assert_equal(tf.shape(unstable_[0]), tf.constant((height, width, 1))),
                                ]):
        stable_ = tf.identity(stable_)
    logger.info('stable_[0].shape={}'.format(stable_[0].shape))
    logger.info('unstable_[0].shape={}'.format(unstable_[0].shape))
    stable_ = tf.concat([stable_[0], stable_[1]], axis=2)
    unstable_ = tf.concat([unstable_[0], unstable_[1]], axis=2)
    flow_ = tf.reshape(tf.sparse_tensor_to_dense(features['flow']), [height, width, -1])[:, :, :2]

    feature_matches1_ = tf.reshape(tf.sparse_tensor_to_dense(features['feature_matches1']), [-1, 4])
    feature_matches2_ = tf.reshape(tf.sparse_tensor_to_dense(features['feature_matches2']), [-1, 4])
    num_matches1_ = tf.shape(feature_matches1_)[0]
    num_matches2_ = tf.shape(feature_matches2_)[0]
    logger.info('feature_matches1_.shape={}, feature_matches2_.shape=q{}'.format(feature_matches1_.shape, feature_matches2_.shape))
    with tf.control_dependencies([tf.assert_less(num_matches1_, tf.constant(max_matches)),
                                tf.assert_less(num_matches2_, tf.constant(max_matches)),
                                ]):
        feature_matches1_ = tf.identity(feature_matches1_)
    feature_matches1_ = tf.pad(feature_matches1_, ((0, max_matches - num_matches1_), (0, 0)))
    feature_matches2_ = tf.pad(feature_matches2_, ((0, max_matches - num_matches2_), (0, 0)))
    feature_matches1_.set_shape([max_matches, 4])
    feature_matches2_.set_shape([max_matches, 4])
    mask1_ = tf.sequence_mask([num_matches1_], max_matches)[0]
    mask2_ = tf.sequence_mask([num_matches2_], max_matches)[0]

    seed = random.randint(0, 2**31 - 1)
    para = get_rand_para(seed) 
    for i in range((before_ch + 1) * 2):
        temp = tf.slice(stable_, [0, 0, i], [-1, -1, 1])
        if (i == 0):
            stable = warp_img(temp, seed, para)
        else:
            stable = tf.concat([stable, warp_img(temp, seed, para)], 2)
    for i in range(after_ch + 2):
        temp = tf.slice(unstable_, [0, 0, i], [-1, -1, 1])
        if (i == 0):
            unstable = warp_img(temp, seed, para)
        else:
            unstable = tf.concat([unstable, warp_img(temp, seed, para)], 2)

    black_mask = get_rand_black_mask()
    x1 = tf.concat([add_mask(tf.slice(stable, [0, 0, 1], [-1, -1, before_ch])), 
                    tf.slice(unstable, [0, 0, 0], [-1, -1, after_ch + 1]),
                    black_mask], 2)
    y1 = tf.slice(stable, [0, 0, 0], [-1, -1, 1])
    x2 = tf.concat([add_mask(tf.slice(stable, [0, 0, before_ch + 2], [-1, -1, before_ch])), 
                    tf.slice(unstable, [0, 0, 1], [-1, -1, after_ch + 1]),
                    black_mask], 2)
    y2 = tf.slice(stable, [0, 0, before_ch + 1], [-1, -1, 1])

    flow = warp_flow(flow_, para)
    feature_matches1, mask1 = warp_point(feature_matches1_, mask1_, para)
    feature_matches2, mask2 = warp_point(feature_matches2_, mask2_, para)
    return x1, y1, x2, y2, flow, feature_matches1, mask1, feature_matches2, mask2

def run():
    x, y = read_and_decode("data/train.tfrecords", 3, False)

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
        logger.info(x_b.shape)
        logger.info(x_b)
        mage_summary = tf.summary.image('y', y_b, 5)
        for i in range(tot_ch):
            temp = tf.slice(x_b, [0, 0, 0, i], [-1, -1, -1, 1])
            mage_summary = tf.summary.image('x' + str(i), temp, 5)
        
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./log/', sess.graph)
        summary_all = sess.run(merged)
        summary_writer.add_summary(summary_all, 0)
        summary_writer.close()

def convert_to_coordinate(pts, width=width, height=height):
    return tuple( ((pts + 1) / 2 * [width, height]).astype(np.int32) )
def test():
    batch_size = 1
    data_x1, data_y1, data_x2, data_y2, data_flow, feature_matches1, mask1, feature_matches2, mask2 = \
        read_and_decode("data9_/test/", 20, False)
       #read_and_decode("/Users/lazycal/workspace/lab/3.1/qudou/data3/test/", 20)

    x1_batch, y1_batch, x2_batch, y2_batch, flow_batch, \
    feature_matches1_batch, mask1_batch, feature_matches2_batch, mask2_batch \
        = tf.train.batch(
            [data_x1, data_y1, data_x2, data_y2, data_flow, feature_matches1, mask1, feature_matches2, mask2],
            batch_size=batch_size)

    sv = tf.train.Supervisor(logdir='./tmp/log', save_summaries_secs=0, saver=None)
    with sv.managed_session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3))) as sess:
        import cv2
        import numpy as np
        import feature_fetcher
        batch_x1s, batch_y1s, batch_x2s, batch_y2s, batch_flows, batch_feature_matches1, batch_mask1, batch_feature_matches2, batch_mask2 = sess.run(
            [x1_batch, y1_batch, x2_batch, y2_batch, flow_batch, feature_matches1_batch, mask1_batch, feature_matches2_batch, mask2_batch])
        unstable = np.tile((batch_x1s[0, :, :, before_ch] + 1)[...,None] / 2 * 255, [1,1,3])
        stable = np.tile((batch_y1s[0, :, :, 0] + 1)[...,None] / 2 * 255, [1,1,3])
        img = np.concatenate([stable, unstable], axis=1)
        gt_matches = feature_fetcher.fetch('6.mp4.avi', 7)
        logger.info('false: ',batch_mask1[0,431:], batch_mask2[0,459:])
        logger.info(gt_matches, batch_feature_matches1)
        for (match, mask) in zip(batch_feature_matches1[0], batch_mask1[0]):
            if not mask: continue
            if np.random.uniform(0, 1) > 0.1: continue
            cv2.line(img, convert_to_coordinate(match[:2]), convert_to_coordinate(match[2:] + [2, 0]), tuple(np.random.rand(3) * 255))
        cv2.imwrite('./test.jpg', img)

        img1 = np.concatenate([cv2.imread('./frames/stable/6/image-0008.jpg'), cv2.imread('./frames/unstable/6/image-0008.jpg')],
                                axis=1)
        logger.info('---------------------------------')
        cvt = lambda x:convert_to_coordinate(x, img1.shape[1] / 2, img1.shape[0])
        for match in gt_matches:
            if np.random.uniform(0, 1) > 0.1: continue
            cv2.line(img1, cvt(match[:2]), cvt(match[2:] + [2, 0]), tuple(np.random.rand(3) * 255))
        cv2.imwrite('./test1.jpg', img1)
if __name__ == '__main__':
    test()
