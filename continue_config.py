import cv2
from PIL import Image
import numpy as np

initial_learning_rate = 1e-4
training_iter = 100000
step_size = 50000
no_theta_iter = 1000000

disp_freq = 50
test_freq = 500
save_freq = 1000
