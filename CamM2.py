import datetime
from math import inf
import numpy as np
import cv2

from MostDifferentFrame import calc_and_draw_hist
from ShapeFilter import valid_shape
from TimeStamp import cut_timestamp, get_boxes


def trigger(h_list, o_list, threshold):
    signal = 0
    if np.array(o_list).any():
        s1_list = []
        s2_list = []
        for i in range(6):
            std1 = np.sum(h_list[i][64:, 0])
            std2 = np.sum(h_list[i][64:, 0])
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


def get_diff(frame, old_frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if old_frame is not None:
        diff = frame.astype(np.int16) - old_frame.astype(np.int16)
        diff = np.abs(diff).astype(np.uint8)
        old_frame = frame.copy()
        frame = diff
    else:
        old_frame = frame.copy()
        diff = frame.astype(np.int16) - old_frame.astype(np.int16)
        diff = np.abs(diff).astype(np.uint8)
        frame = diff
    return old_frame, frame


def get_position(frame, old_list):
    th = 1000
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
    hist_img1, hist1 = calc_and_draw_hist(frame, (0, 0, 255), img_mask)
    # cv2.imshow("hist1", hist_img1)

    img_mask = np.zeros((h, w), np.uint8)
    img_mask[0:y, x1:x2] = 255
    hist_img2, hist2 = calc_and_draw_hist(frame, (0, 0, 255), img_mask)
    # cv2.imshow("hist2", hist_img2)

    img_mask = np.zeros((h, w), np.uint8)
    img_mask[0:y, x2:w] = 255
    hist_img3, hist3 = calc_and_draw_hist(frame, (0, 0, 255), img_mask)
    # cv2.imshow("hist3", hist_img3)

    img_mask = np.zeros((h, w), np.uint8)
    img_mask[y:h, 0:x1] = 255
    hist_img4, hist4 = calc_and_draw_hist(frame, (0, 0, 255), img_mask)
    # cv2.imshow("hist1", hist_img1)

    img_mask = np.zeros((h, w), np.uint8)
    img_mask[y:h, x1:x2] = 255
    hist_img5, hist5 = calc_and_draw_hist(frame, (0, 0, 255), img_mask)
    # cv2.imshow("hist2", hist_img2)

    img_mask = np.zeros((h, w), np.uint8)
    img_mask[y:h, x2:w] = 255
    hist_img6, hist6 = calc_and_draw_hist(frame, (0, 0, 255), img_mask)
    # cv2.imshow("hist3", hist_img3)
    # endregion

    hist_list = [hist1, hist2, hist3, hist4, hist5, hist6]
    p, old, sig = trigger(hist_list, old_list, th)

    old1, old2, old3, old4, old5, old6 = old
    position = p
    return position, [old1, old2, old3, old4, old5, old6]


def start_test_new(
        file_path="Samples\\Sample.mp4",
):
    sample = cv2.VideoCapture(file_path)
    old1 = None
    old2 = None
    old3 = None
    old4 = None
    old5 = None
    old6 = None
    old_frame = None
    current_frame = 0
    while sample.isOpened():
        current_frame += 1

    pass


def start_test_lite(
        src_id,
        file_path="Samples\\Sample.mp4",
        output_path="Outputs",
        file_name="Sample.mp4",
        skip_read=False,
        show_diff=True
):
    file_name = file_name.split(".")[0]

    # region Initialize variables
    old1 = None
    old2 = None
    old3 = None
    old4 = None
    old5 = None
    old6 = None
    old_frame = None
    # url = "http://admin:admin@10.80.84.47:8081"
    # sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    sample = cv2.VideoCapture(file_path)
    # sample = cv2.VideoCapture(url)
    # record = []

    file_count = 0
    next_move = 200
    first_frame = True
    cut_box = [[57, 25, 500]]
    # endregion

    # region Initialize VideoWriter
    fps = sample.get(cv2.CAP_PROP_FPS)
    br = sample.get(cv2.CAP_PROP_BITRATE)
    skip_frame = (int(5 * int(br / 1000)) - 5)
    print("Skipped: ", skip_frame)
    if fps == 0 or fps == inf:
        fps = 15
    size = (
        int(sample.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(sample.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    video_writer = cv2.VideoWriter()
    # endregion

    prev_frames = []
    current_frame = 0
    base_time = datetime.datetime.now()
    now = base_time
    total_time = base_time
    diff_time = base_time
    while sample.isOpened():
        then = datetime.datetime.now()

        current_frame += 1
        total_time += (then - now)
        if current_frame % 500 == 0:
            print(current_frame, '', str(total_time - diff_time))
            print('total use:', ' ', str(total_time - base_time))
            total_time = base_time
            print('diff use:', ' ', str(diff_time - base_time))
            diff_time = base_time

        now = then

        if current_frame % (skip_frame + 1) != 0 and skip_read:
            sample.grab()
            next_move -= 1
            if next_move < 0:
                next_move = 0
                if video_writer.isOpened():
                    video_writer.release()
            end = datetime.datetime.now()
            diff_time += (end - start)
            continue

        ret, frame = sample.read()

        if frame is not None:
            if next_move == 201:
                try:
                    video_writer = cv2.VideoWriter(
                        output_path + "/" + file_name + "_" + str(file_count) + '.mp4',
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        size
                    )
                except Exception as e:
                    print(repr(e))
                file_count += 1
            # frame = enhance(frame)

            # region get Differential Frame
            src_frame = frame.copy()
            # prev_frames = [src_frame]
            prev_frames.append(src_frame)
            if len(prev_frames) > 20:
                prev_frames.pop(0)

            if current_frame % (skip_frame + 1) != 0:

                next_move -= 1
                if next_move < 0:
                    next_move = 0
                    if video_writer.isOpened():
                        video_writer.release()

                if next_move > 0 and video_writer.isOpened():
                    start = datetime.datetime.now()
                    video_writer.write(prev_frames[0])
                    end = datetime.datetime.now()
                    diff_time += (end - start)
                continue

            frame = cv2.resize(frame, (1024, 768))
            if first_frame:
                cut_box = get_boxes(frame)

            frame = cut_timestamp(cut_box=cut_box, vis=frame)

            old_frame, frame = get_diff(frame, old_frame)

            # endregion

            flicker_points = valid_shape(frame)

            old_list = [old1, old2, old3, old4, old5, old6]
            position, old_list = get_position(frame=frame, old_list=old_list)
            old1, old2, old3, old4, old5, old6 = old_list

            if position in [0, 1, 2, 3, 4, 5] and not flicker_points:
                if next_move == 0 or file_count == 0:
                    next_move = 201
                else:
                    next_move = 200
            else:
                next_move -= 1
                if next_move < 0:
                    next_move = 0
                    if video_writer.isOpened():
                        video_writer.release()

            if next_move > 0 and video_writer.isOpened():
                start = datetime.datetime.now()
                video_writer.write(prev_frames[0])
                end = datetime.datetime.now()
                diff_time += (end - start)

            # # region write Rectangles
            # if not flicker_points:
            #     if position == 0:
            #         # src_frame = cv2.rectangle(src_frame, (0, 0), (x1, y), (255, 0, 0), 5, cv2.LINE_AA)
            #         frame = cv2.rectangle(frame, (0, 0), (x1, y), (255, 255, 255), 5, cv2.LINE_AA)
            #     elif position == 1:
            #         # src_frame = cv2.rectangle(src_frame, (x1, 0), (x2, y), (255, 0, 0), 5, cv2.LINE_AA)
            #         frame = cv2.rectangle(frame, (x1, 0), (x2, y), (255, 255, 255), 5, cv2.LINE_AA)
            #     elif position == 2:
            #         # src_frame = cv2.rectangle(src_frame, (x2, 0), (w, y), (255, 0, 0), 5, cv2.LINE_AA)
            #         frame = cv2.rectangle(frame, (x2, 0), (w, y), (255, 255, 255), 5, cv2.LINE_AA)
            #     elif position == 3:
            #         # src_frame = cv2.rectangle(src_frame, (0, y), (x1, h), (255, 0, 0), 5, cv2.LINE_AA)
            #         frame = cv2.rectangle(frame, (0, y), (x1, h), (255, 255, 255), 5, cv2.LINE_AA)
            #     elif position == 4:
            #         # src_frame = cv2.rectangle(src_frame, (x1, y), (x2, h), (255, 0, 0), 5, cv2.LINE_AA)
            #         frame = cv2.rectangle(frame, (x1, y), (x2, h), (255, 255, 255), 5, cv2.LINE_AA)
            #     elif position == 5:
            #         # src_frame = cv2.rectangle(src_frame, (x2, y), (w, h), (255, 0, 0), 5, cv2.LINE_AA)
            #         frame = cv2.rectangle(frame, (x2, y), (w, h), (255, 255, 255), 5, cv2.LINE_AA)
            #     else:
            #         pass
            #
            # if show_diff:
            #     cv2.imshow("diff", frame)
            #     cv2.imshow("origin", cv2.resize(src_frame, (1024, 768)))
            #     # cv2.imshow("cut_ts", inspect_frame)
            #     cv2.waitKey(1)
        else:
            break
        first_frame = False
        # end = datetime.datetime.now()
        # diff_time += (end - start)

    sample.release()
    if video_writer.isOpened:
        video_writer.release()
    cv2.destroyAllWindows()

    return src_id
