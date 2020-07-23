import pickle
import cv2
import os
import sys
import numpy as np
from random import shuffle
from HistUtil import calc_and_draw_hist
from sklearn.metrics.pairwise import cosine_similarity


def get_histogram(file_path=None, image=None):
    if image is not None:
        img = image
    else:
        try:
            assert file_path is not None
            img = cv2.imread(file_path, 1)
        except AssertionError as e:
            print("Either path or image object need to be input")
            sys.exit()
    img_copy = img.copy()
    # region Define sizes
    h = img.shape[0]
    w = img.shape[1]
    x1 = int(0.25 * w)
    x2 = int(0.5 * w)
    x3 = int(0.75 * w)
    y1 = int(0.25 * h)
    y2 = int(0.5 * h)
    y3 = int(0.75 * h)
    xs = int(0.25 * w)
    ys = int(0.25 * h)
    # endregion

    # region Get Histogram
    hist_img_list = []
    hist_row_list = []
    for xa in [0, x1, x2, x3]:
        col_list = []
        for ya in [0, y1, y2, y3]:
            img_mask = np.zeros((h, w), np.uint8)
            img_mask[ya:ya + ys, xa:xa + xs] = 255
            img_copy = cv2.rectangle(img_copy, (xa, ya), (xa + xs, ya + ys), (0, 255, 0), 2, cv2.LINE_AA)
            hist_img, hist = calc_and_draw_hist(img, (0, 0, 255), img_mask)
            hist_img_list.append(hist_img)
            col_list.append(hist)
        hist_row_list.append(col_list)
    # endregion

    hist_array = np.squeeze(np.array(hist_row_list))
    return hist_array, img_copy


def save_template(hist_array):
    temp = open("4X4_Template_Histograms.pkl", "wb")
    pickle.dump(hist_array, temp)
    temp.close()


def cal_cos_similarity(temp_hist, hist_array):
    result = np.zeros(shape=(temp_hist.shape[0], temp_hist.shape[1]))
    for i in range(temp_hist.shape[0]):
        for j in range(temp_hist.shape[1]):
            temp = temp_hist[i, j, :]
            sample = hist_array[i, j, :]
            cs = cosine_similarity(temp.reshape(1, -1), sample.reshape(1, -1))
            result[i, j] = np.squeeze(cs)
    return result


def test_loop():
    path = "Samples"
    temp_hist = None
    file_list = os.listdir(path)
    shuffle(file_list)
    for i in range(len(file_list)):
        if "(1)" in file_list[i]:
            temp = file_list[0]
            file_list[0] = file_list[i]
            file_list[1] = temp
            break

    for e, i in enumerate(file_list):
        if ("blur" not in i) and i.endswith(".jpg"):
            hist_array, img = get_histogram(os.path.join(path, i))
            img = cv2.resize(img, (512, 384))
            if "(1)" in i:
                save_template(hist_array)
                temp_hist = hist_array
                cv2.imshow("template", img)
            if temp_hist is not None:
                cs_list = cal_cos_similarity(temp_hist, hist_array)
                cs_list = cs_list[1:3, :]
                cs_list_bin = cs_list.copy()
                cs_list_bin[np.where(cs_list_bin <= 0.75)] = 0
                valid_area_count = np.count_nonzero(cs_list_bin)
                print(i)
                print(cs_list_bin.T)
                print(valid_area_count)
                if valid_area_count >= 5:
                    cv2.imshow("positive sample", img)
                else:
                    cv2.imshow("negative sample", img)
                cv2.waitKey()


# test_loop()
