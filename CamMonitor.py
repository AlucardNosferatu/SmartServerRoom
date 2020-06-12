import os
import sys
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt


def enhance(f):
    # 线性变换
    a = 2
    o = float(a) * f
    o -= 50
    o[o > 255] = 255  # 大于255要截断为255
    o[o < 0] = 0
    # 数据类型的转换
    o = np.round(o)
    o = o.astype(np.uint8)
    return o


def calcAndDrawHist(image, color, mask=None):
    hist = cv2.calcHist([image], [0], mask, [256], [0.0, 255.0])
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
    hist_img = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)
    for h in range(256):
        intensity = int(hist[h] * hpt / max_val)
        cv2.line(hist_img, (h, 256), (h, 256 - intensity), color)
    return hist_img, hist


def trigger(h_list, o_list, f_list, threshold, use_diff):
    signal = 0
    if np.array(o_list).any():
        s1_list = []
        s2_list = []
        for i in range(6):
            if use_diff:
                std1 = np.sum(h_list[i][64:, 0])
                std2 = np.sum(h_list[i][64:, 0])
            else:
                std1 = np.sqrt(np.sum(np.power(h_list[i] - o_list[i], 2)))
                std2 = np.sqrt(np.sum(np.power(h_list[i] - f_list[i], 2)))
            s1_list.append(std1)
            s2_list.append(std2)
        s3_list = np.multiply(np.array(s1_list), np.array(s2_list)).tolist()
        o_list = h_list
        signal = max(s3_list)
        # print(signal)
        if signal < threshold:
            pos = -1
        else:
            pos = s3_list.index(signal)
    else:
        o_list = h_list
        pos = -1

    return pos, o_list, signal


def start_test(show_diff=False, file_path="Samples\\Sample.mp4", output_path="Outputs", file_name="Sample.mp4"):
    file_name = file_name.split(".")[0]
    # region Initialize variables
    old1 = None
    old2 = None
    old3 = None
    old4 = None
    old5 = None
    old6 = None
    first1 = None
    first2 = None
    first3 = None
    first4 = None
    first5 = None
    first6 = None
    record = []
    old_frame = None
    # url = "http://admin:admin@10.80.84.47:8081"
    # sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    sample = cv2.VideoCapture(file_path)
    # sample = cv2.VideoCapture(url)
    th_line = []
    record = []
    count = 0
    th = 2e5
    use_diff = True
    file_count = 0
    next_move = 100
    # endregion

    # plt.ion()  # 开启interactive mode 成功的关键函数
    # plt.figure(1)

    # region Initialize VideoWriter
    fps = sample.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 15
    size = (
        int(sample.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(sample.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    video_writer = None
    # endregion

    while sample.isOpened():
        sig = 0
        plt.clf()
        position = -1
        ret, frame = sample.read()
        th_line = [th] * 200
        if frame is not None:
            print(next_move)
            if next_move == 101:
                print("Start")
                video_writer = cv2.VideoWriter(
                    output_path + "\\" + file_name + "_" + str(file_count) + '.avi',
                    cv2.VideoWriter_fourcc(*'MJPG'),
                    fps,
                    size
                )
                file_count += 1
            # frame = enhance(frame)
            # region get Differential Frame
            src_frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if old_frame is not None and use_diff:
                diff = frame.astype(np.int16) - old_frame.astype(np.int16)
                diff = np.abs(diff).astype(np.uint8)
                old_frame = frame.copy()
                frame = diff
            elif use_diff:
                old_frame = frame.copy()
                diff = frame.astype(np.int16) - old_frame.astype(np.int16)
                diff = np.abs(diff).astype(np.uint8)
                frame = diff
            # endregion

            # region get Sizes
            h = frame.shape[0]
            w = frame.shape[1]
            x1 = int(0.3 * w)
            x2 = int(0.7 * w)
            y = int(0.5 * h)
            # endregion

            # region get Histogram
            img_mask = np.zeros((h, w), np.uint8)
            img_mask[0:y, 0:x1] = 255
            hist_img1, hist1 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
            # cv2.imshow("hist1", hist_img1)

            img_mask = np.zeros((h, w), np.uint8)
            img_mask[0:y, x1:x2] = 255
            hist_img2, hist2 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
            # cv2.imshow("hist2", hist_img2)

            img_mask = np.zeros((h, w), np.uint8)
            img_mask[0:y, x2:w] = 255
            hist_img3, hist3 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
            # cv2.imshow("hist3", hist_img3)

            img_mask = np.zeros((h, w), np.uint8)
            img_mask[y:h, 0:x1] = 255
            hist_img4, hist4 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
            # cv2.imshow("hist1", hist_img1)

            img_mask = np.zeros((h, w), np.uint8)
            img_mask[y:h, x1:x2] = 255
            hist_img5, hist5 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
            # cv2.imshow("hist2", hist_img2)

            img_mask = np.zeros((h, w), np.uint8)
            img_mask[y:h, x2:w] = 255
            hist_img6, hist6 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
            # cv2.imshow("hist3", hist_img3)

            # endregion
            count += 1
            first_hist = first1 is None and first2 is None and first3 is None and first4 is None and first5 is None and first6 is None
            if first_hist or count > 200:
                count = 0
                first1 = hist1
                first2 = hist2
                first3 = hist3
                first4 = hist4
                first5 = hist5
                first6 = hist6

            hist_list = [hist1, hist2, hist3, hist4, hist5, hist6]
            old_list = [old1, old2, old3, old4, old5, old6]
            first_list = [first1, first2, first3, first4, first5, first6]
            p, old, sig = trigger(hist_list, old_list, first_list, th, use_diff)
            old1, old2, old3, old4, old5, old6 = old

            if sig > 2 * th:
                sig = 2 * th
            record.append(sig)
            if len(record) > 200:
                del record[0]
            # plt.plot(record)
            # plt.plot(th_line)
            # plt.pause(0.005)
            position = p

            if position in [0, 1, 2, 3, 4, 5]:
                if next_move == 0 or file_count == 0:
                    next_move = 101
                else:
                    next_move = 100
            else:
                next_move -= 1
                if next_move < 0:
                    next_move = 0
                    if video_writer:
                        video_writer.release()

            if next_move > 0 and video_writer:
                print("Recording")
                video_writer.write(src_frame)

            # region write Rectangles
            if position == 0:
                # src_frame = cv2.rectangle(src_frame, (0, 0), (x1, y), (255, 0, 0), 5, cv2.LINE_AA)
                frame = cv2.rectangle(frame, (0, 0), (x1, y), (255, 255, 255), 5, cv2.LINE_AA)
            elif position == 1:
                # src_frame = cv2.rectangle(src_frame, (x1, 0), (x2, y), (255, 0, 0), 5, cv2.LINE_AA)
                frame = cv2.rectangle(frame, (x1, 0), (x2, y), (255, 255, 255), 5, cv2.LINE_AA)
            elif position == 2:
                # src_frame = cv2.rectangle(src_frame, (x2, 0), (w, y), (255, 0, 0), 5, cv2.LINE_AA)
                frame = cv2.rectangle(frame, (x2, 0), (w, y), (255, 255, 255), 5, cv2.LINE_AA)
            elif position == 3:
                # src_frame = cv2.rectangle(src_frame, (0, y), (x1, h), (255, 0, 0), 5, cv2.LINE_AA)
                frame = cv2.rectangle(frame, (0, y), (x1, h), (255, 255, 255), 5, cv2.LINE_AA)
            elif position == 4:
                # src_frame = cv2.rectangle(src_frame, (x1, y), (x2, h), (255, 0, 0), 5, cv2.LINE_AA)
                frame = cv2.rectangle(frame, (x1, y), (x2, h), (255, 255, 255), 5, cv2.LINE_AA)
            elif position == 5:
                # src_frame = cv2.rectangle(src_frame, (x2, y), (w, h), (255, 0, 0), 5, cv2.LINE_AA)
                frame = cv2.rectangle(frame, (x2, y), (w, h), (255, 255, 255), 5, cv2.LINE_AA)
            else:
                pass
            # endregion
            if show_diff:
                cv2.imshow("diff", frame)
                cv2.imshow("origin", src_frame)
        else:
            break

        # region key interface
        k = cv2.waitKey(50)
        if k & 0xff == ord('q'):
            print("redefine threshold")
            th = th * 1.05
        if k & 0xff == ord('e'):
            print("redefine threshold")
            th = th * 0.95
        if k & 0xff == ord('w'):
            break
        # endregion
    sample.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


def process_dir(dir_path="Samples", output_path="Outputs"):
    for e, i in enumerate(os.listdir(dir_path)):
        if i.endswith('mp4') or i.endswith('MP4'):
            file_path = os.path.join(dir_path, i)
            start_test(True, file_path, output_path, i)
            # if os.path.exists(os.path.join(dir_path, i)):
            #     os.remove(os.path.join(dir_path, i))
            #     print("src video file has been deleted")


process_dir()
# process_dir(sys.argv[1], sys.argv[2])
