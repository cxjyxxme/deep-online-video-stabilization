import tensorflow as tf
import numpy as np
from config import *
from PIL import Image
import cv2
import time

model_name = 'model-44000'
new_saver = tf.train.import_meta_graph('models/' + model_name + '.meta')
graph = tf.get_default_graph()

x_tensor = graph.get_tensor_by_name('input/x_tensor:0')
x_batch = graph.get_tensor_by_name('datas/shuffle_batch:0')
y_batch = graph.get_tensor_by_name('datas/shuffle_batch:1')
theta_loss = graph.get_tensor_by_name('theta_loss/mul:0')
total_loss = graph.get_tensor_by_name('add_1:0')
merged = graph.get_tensor_by_name('Merge/MergeSummary:0')
y = graph.get_tensor_by_name('label/Placeholder:0')
use_theta_loss = graph.get_tensor_by_name('theta_loss/Placeholder:0')
test_x_batch = graph.get_tensor_by_name('datas/shuffle_batch_1:0')
test_y_batch = graph.get_tensor_by_name('datas/shuffle_batch_1:1')
test_merged = graph.get_tensor_by_name('Merge_1/MergeSummary:0')
loss_displayer = graph.get_tensor_by_name('Placeholder:0')

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=step_size,decay_rate=0.1, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate, name='new_optimizer')
optimizer = opt.minimize(total_loss, global_step=global_step)
init = tf.initialize_all_variables()
saver = tf.train.Saver()
sv = tf.train.Supervisor(logdir='temp/log', save_summaries_secs=0, saver=None)
with sv.managed_session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95))) as sess:
    #??? threads = tf.train.start_queue_runners(sess=sess)
    sess.run(init)
    new_saver.restore(sess, 'models/' + model_name)
    tf.train.start_queue_runners(sess=sess)

    time_start = time.time()
    tot_time = 0

    for i in range(training_iter):
        batch_xs, batch_ys = sess.run([x_batch, y_batch])
        if (i > no_theta_iter):
            use_theta = 0
        else:
            use_theta = 1
        if i % disp_freq == 0:
            print('time:' + str(tot_time) + 's')
            tot_time = 0
            time_start = time.time()
            t_loss, loss, summary = sess.run([theta_loss, total_loss, merged],
                            feed_dict={
                                x_tensor: batch_xs,
                                y: batch_ys,
                                use_theta_loss: use_theta
                            })
            sv.summary_writer.add_summary(summary, i)
            print('Iteration: ' + str(i) + ' Loss: ' + str(loss) + ' ThetaLoss: ' + str(t_loss*theta_mul))
            lr = sess.run(learning_rate)
            print(lr)
            time_end = time.time()
            print('disp time:' + str(time_end - time_start) + 's')
        if i % test_freq == 0:
            sum_test_loss = 0.0
            for j in range(test_batches):
                test_batch_xs, test_batch_ys = sess.run([test_x_batch, test_y_batch])
                loss = sess.run(total_loss,
                            feed_dict={
                                x_tensor: test_batch_xs,
                                y: test_batch_ys,
                                use_theta_loss: use_theta
                            })
                sum_test_loss += loss
            sum_test_loss /= test_batches
            print("Test Loss: " + str(sum_test_loss))
            summary = sess.run(test_merged,
                    feed_dict={
                        loss_displayer: sum_test_loss
                    })
            sv.summary_writer.add_summary(summary, i)
        if i % save_freq == 0:
            saver.save(sess, 'temp/models/model', global_step=i)
        time_end = time.time()
        tot_time += time_end - time_start
        sess.run(optimizer, feed_dict={
            x_tensor: batch_xs, y: batch_ys, use_theta_loss: use_theta})   
        time_start = time.time()

