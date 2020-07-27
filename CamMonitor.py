import os
from math import inf

import cv2
import numpy as np

from HTTPInterface import post_result, MyRequestHandler, HTTPServer
from MostDifferentFrame import snap_shot
from ShapeFilter import valid_shape
from TimeStamp import get_boxes, cut_timestamp


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


def calc_and_draw_hist(image, color, mask=None):
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


def start_test(
        src_id,
        show_diff=False,
        file_path="Samples\\Sample.mp4",
        output_path="Outputs",
        file_name="Sample.mp4",
        skip_frame=5
):
    file_name = file_name.split(".")[0]
    print(file_path)
    print(output_path)
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
    old_frame = None
    # url = "http://admin:admin@10.80.84.47:8081"
    # sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    sample = cv2.VideoCapture(file_path)
    # sample = cv2.VideoCapture(url)
    record = []
    th = 1e3
    use_diff = True
    file_count = 0
    next_move = 100
    first_frame = True
    cut_box = [[57, 25, 500]]
    # endregion

    # region Initialize VideoWriter
    fps = sample.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps == inf:
        fps = 15
    size = (
        int(sample.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(sample.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    video_writer = cv2.VideoWriter()
    # endregion
    count = 0
    prev_frames = []
    while sample.isOpened():
        ret, frame = sample.read()
        if frame is not None:
            # print(next_move)
            if next_move == 101:
                print("Start")
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
            prev_frames.append(src_frame)
            if len(prev_frames) > 20:
                prev_frames.pop(0)
            if count % (skip_frame + 1) != 0:
                count += 1
                next_move -= 1
                if next_move < 0:
                    next_move = 0
                    if video_writer.isOpened():
                        video_writer.release()
                if next_move > 0 and video_writer.isOpened():
                    # print("Recording")
                    video_writer.write(prev_frames[0])
                continue
            frame = cv2.resize(frame, (1024, 768))
            if first_frame:
                cut_box = get_boxes(frame)
            frame = cut_timestamp(cut_box=cut_box, vis=frame)
            inspect_frame = frame.copy()
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

            flicker_points = valid_shape(frame)

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

            count += 1
            first_hist = first1 is None
            first_hist = first_hist and first2 is None
            first_hist = first_hist and first3 is None
            first_hist = first_hist and first4 is None
            first_hist = first_hist and first5 is None
            first_hist = first_hist and first6 is None
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
            position = p

            if position in [0, 1, 2, 3, 4, 5] and not flicker_points:
                if next_move == 0 or file_count == 0:
                    next_move = 101
                else:
                    next_move = 100
            else:
                next_move -= 1
                if next_move < 0:
                    next_move = 0
                    if video_writer.isOpened():
                        video_writer.release()

            if next_move > 0 and video_writer.isOpened():
                # print("Recording")
                video_writer.write(prev_frames[0])

            # region write Rectangles
            if not flicker_points:
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
                cv2.imshow("origin", cv2.resize(src_frame, (1024, 768)))
                cv2.imshow("cut_ts", inspect_frame)
                # cv2.waitKey()
        else:
            break
        first_frame = False
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
    if video_writer.isOpened:
        video_writer.release()
    cv2.destroyAllWindows()
    return src_id


def process_dir(
        _,
        request_id,
        dir_path="C:\\Users\\16413\\Desktop\\FFCS\\SVN\\CV_Toolbox\\SmartServerRoom\\Samples",
        output_path="C:\\Users\\16413\\Desktop\\FFCS\\SVN\\CV_Toolbox\\SmartServerRoom\\Outputs"
):
    indices = range(245, 289)
    print("before start_test: ", request_id)
    if type(dir_path) == list:
        dir_path = dir_path[0]
    if type(output_path) == list:
        output_path = output_path[0]
    src_num = 0
    dst_num = 0
    src_id = 0
    for e, i in enumerate(os.listdir(dir_path)):
        if (i.endswith('mp4') or i.endswith('MP4')) and True:
            file_path = os.path.join(dir_path, i)
            print('file_path is:', file_path)
            src_id = start_test(
                show_diff=True,
                # show_diff=False,
                file_path=file_path,
                output_path=output_path,
                file_name=i,
                src_id=request_id
            )
            src_num += 1
    out_files = os.listdir(output_path)
    for e, i in enumerate(out_files):
        if i.endswith('mp4') and True:
            file_path = os.path.join(output_path, i)
            snap_shot(calc_and_draw_hist, file_path=file_path)
            dst_num += 1
            # print(os.listdir(output_path))
    assert src_id == request_id
    print("after start_test: ", request_id)
    post_result(request_id, src_num, dst_num)


def delete_file(dir_path, i):
    if os.path.exists(os.path.join(dir_path, i)):
        os.remove(os.path.join(dir_path, i))
        print("src video file has been deleted")


def start_server():
    MyRequestHandler.process = process_dir
    server = HTTPServer(("", 5673), MyRequestHandler)
    print("Serving at http://localhost:5673/imr-monitor-server/parsevideo")
    server.serve_forever()


def specify_index(indices, i):
    for index in indices:
        if '_' + str(index) in i:
            return True
    return False


if __name__ == '__main__':
    # start_server()
    process_dir(_=None, request_id='1')
