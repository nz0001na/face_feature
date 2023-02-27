import csv
import cv2
import numpy as np
import scipy.io as sio
import radialProfile
import os

ro = '/media/zn/Elements/1_research/1_MAD/1_Data/'
img_path_b = ro + '4_StyleGAN_Morphs_data/6_cropped_resize/'
img_path_m = ro + '2_AMSL_FFHQ_cropped/londondb_morph_combined_alpha0.5_passport-scale_15kb/'
# img_path_m = ro + '3_AMSL_FFHQ_cropped_grayscale_resize/size_270_270/londondb_morph_combined_alpha0.5_passport-scale_15kb'
#
num = ['N200','N2000']
for t in range(len(num)):
    NUM = num[t]
    dst_path = ro + '8_texture_feature/99_train_test_AMSL_aug/'+ NUM+'/'
    if os.path.exists(dst_path) is False:
        os.makedirs(dst_path)

    list_path = ro + '100_detection_experiment/1_train_test_list/4_AMSL_aug/'
    train_file = list_path + NUM + '_train_list.csv'
    test_file = list_path + NUM + '_test_list.csv'

    # read train feature
    train_feature = []
    train_label = []
    train_feature2 = []
    train_label2 = []
    f = csv.reader(open(train_file, 'rb'))
    for row in f:
        ll = str(row[1])
        if ll == '1':
            img_path = img_path_b
        if ll == '0':
            img_path = img_path_m

        fi = img_path + row[0]
        img = cv2.imread(fi, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(270, 270))
        # FFT transformation
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        fshift += 1e-8
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        # Calculate the azimuthally averaged 1D power spectrum
        psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

        train_feature.append(magnitude_spectrum.flatten())
        train_label.append(int(row[1]))
        train_feature2.append(psd1D.flatten())
        train_label2.append(int(row[1]))

    test_feature = []
    test_label = []
    test_feature2 = []
    test_label2 = []
    f = csv.reader(open(test_file, 'rb'))
    for row in f:
        ll = str(row[1])
        if ll == '1':
            img_path = img_path_b
        if ll == '0':
            img_path = img_path_m

        fi = img_path + row[0]
        img = cv2.imread(fi, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (270, 270))
        # FFT transformation
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        fshift += 1e-8
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        # Calculate the azimuthally averaged 1D power spectrum
        psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

        test_feature.append(magnitude_spectrum.flatten())
        test_label.append(int(row[1]))

        test_feature2.append(psd1D.flatten())
        test_label2.append(int(row[1]))

    data = {}
    data["train_feature"] = train_feature
    data["train_label"] = train_label
    data["test_feature"] = test_feature
    data["test_label"] = test_label
    sio.savemat(dst_path + 'DFT_train_test.mat', mdict=data)

    data2 = {}
    data2["train_feature"] = train_feature2
    data2["train_label"] = train_label2
    data2["test_feature"] = test_feature2
    data2["test_label"] = test_label2
    sio.savemat(dst_path + 'FFT_train_test.mat', mdict=data)
