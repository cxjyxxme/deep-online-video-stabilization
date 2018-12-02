import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np

checkpoint_file = 'data_video/resnet_v1_50.ckpt'

inputs = tf.placeholder(tf.float32, [None, 288, 512, 15])
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    net, end_points = resnet_v2.resnet_v2_50(inputs, global_pool = False, is_training=True, output_stride=32)
print(end_points)
merged = tf.summary.merge_all()
init_all = tf.initialize_all_variables()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./log/test/', sess.graph)
    writer.flush()
'''
sample_images = ['dog.jpg', 'panda.jpg']
#Load the model
sess = tf.Session()
arg_scope = inception_resnet_v2_arg_scope()
with slim.arg_scope(arg_scope):
  logits, end_points = inception_resnet_v2(input_tensor, is_training=False)
saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)
for image in sample_images:
  im = Image.open(image).resize((299,299))
  im = np.array(im)
  im = im.reshape(-1,299,299,3)
  predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})
  print (np.max(predict_values), np.max(logit_values))
  print (np.argmax(predict_values), np.argmax(logit_values))
'''
