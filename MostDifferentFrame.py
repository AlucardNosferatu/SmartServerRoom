import datetime
import os

import cv2
import numpy as np
from avtk.backends.ffmpeg.shortcuts import convert_to_h264

from utils import upload


def snap_atom(calc_and_draw_hist, file_path="Outputs/010tMonitorCollect202007190100000150002fc14e_100_0.mp4"):
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
        return None, None
    else:
        base_time = datetime.datetime.now()
        total_time = base_time
        for i in range(hists.shape[0] - 1):
            this_time = hists[i, :]
            now = datetime.datetime.now()

            ed_many = np.linalg.norm(this_time - hists[i + 1:, :], axis=-1)
            ed = np.max(ed_many)
            if ed > max_ed:
                max_ed = ed
                max_i = i
                max_j = np.argmax(ed_many) + i + 1
            then = datetime.datetime.now()

            # for j in range(i + 1, hists.shape[0]):
            #     that_time = hists[j, :]
            #     ed = np.linalg.norm(this_time - that_time)
            #     # print("ed is: ", ed)
            #     if ed > max_ed:
            #         max_ed = ed
            #         max_i = i
            #         max_j = j
            # then = datetime.datetime.now()

            total_time += (then - now)
        total_time -= base_time
        # print('time used: ', str(total_time))
        # print(max_i)
        # print(max_j)
        sample = cv2.VideoCapture(file_path)
        sample.set(1, max_i)
        ret, frame_a = sample.read()
        sample.set(1, max_j)
        ret, frame_b = sample.read()
        sample.release()
        cv2.destroyAllWindows()
        return frame_a, frame_b


def snap_shot(calc_and_draw_hist, file_path="Outputs/010tMonitorCollect202007190100000150002fc14e_100_0.mp4"):
    frame_a, frame_b = snap_atom(calc_and_draw_hist, file_path)
    result_dict = {}
    if frame_a is not None and frame_b is not None:
        # cv2.imshow("A", cv2.resize(frame_A, (int(frame_A.shape[1] / 4), int(frame_A.shape[0] / 4))))
        # cv2.imwrite(file_path.replace('.mp4', '_A.jpg'), frame_a)
        cv2.imencode('.jpg', frame_a)[1].tofile(file_path.replace('.mp4', '_A.jpg'))
        result_dict['a'] = upload(file_name=file_path.replace('.mp4', '_A.jpg'), to_temp=False, deletion=False,
                                  file_dir='')
        # cv2.imshow("B", cv2.resize(frame_B, (int(frame_B.shape[1] / 4), int(frame_B.shape[0] / 4))))
        # cv2.imwrite(file_path.replace('.mp4', '_B.jpg'), frame_b)
        cv2.imencode('.jpg', frame_b)[1].tofile(file_path.replace('.mp4', '_B.jpg'))
        result_dict['b'] = upload(file_name=file_path.replace('.mp4', '_B.jpg'), to_temp=False, deletion=False,
                                  file_dir='')
        # cv2.waitKey()

        # print("Start conversion.")
        convert_to_h264(file_path, file_path + ".new")
        # print("Conversion has been completed.")
        os.remove(file_path)
        # print("Src video removed.")
        os.rename(file_path + '.new', file_path)
        result_dict['mp4'] = upload(file_name=file_path, to_temp=False, deletion=False, file_dir='')
    return result_dict
    # print("New Video renamed.")


def calc_and_draw_hist(image, color, mask=None):
    hist = cv2.calcHist([image], [0], mask, [256], [0.0, 255.0])
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
    hist_img = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)
    for h in range(256):
        intensity = int(hist[h] * hpt / max_val)
        cv2.line(hist_img, (h, 256), (h, 256 - intensity), color)
    return hist_img, hist


if __name__ == '__main__':
    snap_shot(
        calc_and_draw_hist=calc_and_draw_hist,
        file_path='Outputs/010tMonitorCollect202007141118363150016f2102_110_1.mp4'
    )
