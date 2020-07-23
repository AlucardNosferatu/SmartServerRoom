import os
import time

from avtk.backends.ffmpeg.shortcuts import convert_to_h264
import cv2
import numpy as np


def snap_shot(calc_and_draw_hist, file_path="Outputs/010tMonitorCollect202007190100000150002fc14e_100_0.mp4"):
    sample = cv2.VideoCapture(file_path)
    hist_list = []
    while sample.isOpened():
        ret, frame = sample.read()
        if frame is not None:
            hist_img, hist = calc_and_draw_hist(frame, (0, 0, 255), None)
            hist_list.append(hist.tolist())
        else:
            break
    sample.release()
    cv2.destroyAllWindows()

    hists = np.squeeze(np.array(hist_list))
    max_ed = 0
    max_i = 0
    max_j = 0
    if len(hists.shape) < 2:
        os.remove(file_path)
    else:
        for i in range(hists.shape[0]):
            this_time = hists[i, :]
            for j in range(i + 1, hists.shape[0]):
                that_time = hists[j, :]
                ed = np.linalg.norm(this_time - that_time)
                # print("ed is: ", ed)
                if ed > max_ed:
                    max_ed = ed
                    max_i = i
                    max_j = j
        print(max_i)
        print(max_j)
        sample = cv2.VideoCapture(file_path)
        sample.set(1, max_i)
        ret, frame_a = sample.read()
        sample.set(1, max_j)
        ret, frame_b = sample.read()
        # cv2.imshow("A", cv2.resize(frame_A, (int(frame_A.shape[1] / 4), int(frame_A.shape[0] / 4))))
        cv2.imwrite(file_path.replace('.mp4', '_A.jpg'), frame_a)
        # cv2.imshow("B", cv2.resize(frame_B, (int(frame_B.shape[1] / 4), int(frame_B.shape[0] / 4))))
        cv2.imwrite(file_path.replace('.mp4', '_B.jpg'), frame_b)
        # cv2.waitKey()
        sample.release()
        cv2.destroyAllWindows()
        # print("Start conversion.")
        convert_to_h264(file_path, file_path + ".new")
        # print("Conversion has been completed.")
        os.remove(file_path)
        # print("Src video removed.")
        os.rename(file_path + '.new', file_path)
        # print("New Video renamed.")
