'''
This code is used to extract LBP feature
'''

import skimage.feature as sf
from PIL import Image
import scipy.io as sio
import os
import csv
import numpy as np
import cv2

# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

ro = '/home/na/1_MAD/2_Data/4_AMSL_aug_all/'
img_path = ro + 'neutral_aug_folder/'

dst_path = ro + '8_texture_feature/train_test_AMSL_aug_all/'
if os.path.exists(dst_path) is False:
    os.makedirs(dst_path)

list_path = ro + '100_detection_experiment/1_train_test_list/'
train_file = list_path + 'all_train_list.csv'
test_file = list_path + 'all_test_list.csv'

# read train feature
train_feature = []
train_label = []
f = csv.reader(open(train_file, 'r'))
for row in f:
    fi = img_path + row[0]
    image = cv2.imread(fi, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (270, 270))
    lbp = sf.local_binary_pattern(image, n_points, radius, METHOD)
    train_feature.append(lbp.flatten())
    train_label.append(int(row[1]))

# read test feature
test_feature = []
test_label = []
f = csv.reader(open(test_file, 'r'))
for row in f:
    fi = img_path + row[0]
    # Open image, input image: color
    # image = Image.open(fi)
    image = cv2.imread(fi, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (270, 270))
    lbp = sf.local_binary_pattern(image, n_points, radius, METHOD)
    test_feature.append(lbp.flatten())
    test_label.append(int(row[1]))

dict = {}
dict['train_feature'] = np.asarray(train_feature)
dict['train_label'] = np.asarray(train_label).flatten()
dict['test_feature'] = np.asarray(test_feature)
dict['test_label'] = np.asarray(test_label).flatten()
dst_file = dst_path + 'LBP_train_test.mat'
sio.savemat(dst_file, mdict=dict)

