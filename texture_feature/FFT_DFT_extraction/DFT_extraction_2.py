'''
This code is to
(1) use DFT to calculate frequency spectrum on cropped face images
'''

import os
import csv
import cv2
import scipy.io as sio
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
import gc

ro = '/home/na/1_MAD/2_Data/4_AMSL_aug_all/'
src_path = ro + 'neutral_aug_folder/'

dst_path = ro + '8_texture_feature/DFT/'
if os.path.exists(dst_path) is False:
    os.makedirs(dst_path)

dst_dftimg_path = dst_path + 'DFT_fig/'
dst_dftmat_path = dst_path + 'DFT_mat/'
if os.path.exists(dst_dftimg_path) is False:
    os.makedirs(dst_dftimg_path)
if os.path.exists(dst_dftmat_path) is False:
    os.makedirs(dst_dftmat_path)

img_folder_list = os.listdir(src_path)
for j in range(len(img_folder_list)): # len(img_folder_list)
    img_folder = img_folder_list[j]
    if img_folder != 'londondb_morph_combined_alpha0.5_passport-scale_15kb':
        continue
    print(img_folder)
    if os.path.exists(dst_dftimg_path + img_folder + '/') is False:
        os.makedirs(dst_dftimg_path + img_folder + '/')
    if os.path.exists(dst_dftmat_path + img_folder + '/') is False:
        os.makedirs(dst_dftmat_path + img_folder + '/')

    cell_list = os.listdir(src_path + img_folder + '/')
    for k in range(len(cell_list)):
        cell_name = cell_list[k]
        name = cell_name[:len(cell_name)-4]

        img = cv2.imread(src_path + img_folder + '/' + cell_name, 0)
        img = cv2.resize(img, (270, 270), interpolation=cv2.INTER_AREA)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        # save data
        savemat(dst_dftmat_path + img_folder + '/' + name + '.mat', {'magnitude_spectrum': magnitude_spectrum})
        plt.imshow(magnitude_spectrum, cmap = 'inferno')
        plt.savefig(dst_dftimg_path + img_folder + '/' + name + '.jpg', bbox_inches='tight')
        plt.close()
        gc.collect()






