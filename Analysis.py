import datetime
from math import inf
import numpy as np
import cv2

from MostDifferentFrame import calc_and_draw_hist
from ShapeFilter import valid_shape
from TimeStamp import cut_timestamp, get_boxes


def trigger(h_list, threshold):
    s1_list = []
    s2_list = []
    for i in range(6):
        std1 = np.sum(h_list[i][64:, 0])
        std2 = np.sum(h_list[i][64:, 0])
        s1_list.append(std1)
        s2_list.append(std2)
    s3_list = np.multiply(np.array(s1_list), np.array(s2_list)).tolist()
    signal = max(s3_list)
    if signal < threshold:
        pos = -1
    else:
        pos = s3_list.index(signal)
    return pos, signal


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


def get_position(frame, sizes):
    # region get Sizes
    th = 1000
    w, h = sizes
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
    p, sig = trigger(hist_list, th)
    return p


def mark_motion(new_size, position, frame, src_frame, flicker_points=False, show_diff=True):
    # region write Rectangles
    w, h = new_size
    x1 = int(0.3 * w)
    x2 = int(0.7 * w)
    y = int(0.5 * h)
    if not flicker_points and show_diff:
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
        cv2.imshow("diff", frame)
        cv2.imshow("origin", cv2.resize(src_frame, (1024, 768)))
        cv2.waitKey()


def start_test_lite(
        src_id,
        file_path="Samples\\Sample.mp4",
        output_path="Outputs",
        file_name="Sample.mp4",
        skip_read=False,
        show_diff=False
):
    file_name = file_name.split(".")[0]
    old_frame = None
    sample = cv2.VideoCapture(file_path)
    file_count = 0
    next_move = 200
    first_frame = True
    cut_box = [[57, 25, 500]]

    # region Initialize VideoWriter
    fps = sample.get(cv2.CAP_PROP_FPS)
    br = sample.get(cv2.CAP_PROP_BITRATE)
    skip_frame = int(5 * int(br / 1000))
    print("Skipped: ", skip_frame)
    if fps == 0 or fps == inf:
        fps = 15
    size = (
        int(sample.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(sample.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    new_size = (1024, 768)
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

            frame = cv2.resize(frame, new_size)
            if first_frame:
                cut_box = get_boxes(frame)

            frame = cut_timestamp(cut_box=cut_box, vis=frame)

            old_frame, frame = get_diff(frame, old_frame)
            # endregion

            flicker_points = valid_shape(frame)

            position = get_position(
                frame=frame,
                sizes=new_size
            )

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

            mark_motion(
                new_size=new_size,
                position=position,
                frame=frame,
                src_frame=src_frame,
                flicker_points=flicker_points,
                show_diff=show_diff
            )
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


def start_test_new(
        src_id,
        file_path="Samples\\Sample.mp4",
        output_path="Outputs",
        file_name='Sample.mp4',
        skip_frame=100
):
    file_name = file_name.split(".")[0]
    cut_box = [[57, 25, 500]]
    new_size = (1024, 768)
    sample = cv2.VideoCapture(file_path)
    old_frame = None
    current_frame = -1
    skip_read = True
    record_now = False
    vw = None
    fps = 15
    size = (
        int(sample.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(sample.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    file_count = 0
    while sample.isOpened():
        current_frame = int(sample.get(cv2.CAP_PROP_POS_FRAMES))
        if skip_read:
            if record_now:
                print('Error! Cannot record during skip mode!')
            else:
                print('Skip Mode')
        else:
            if record_now:
                print('Record Mode')
            else:
                print('Rewind Mode')
        if current_frame % (skip_frame + 1) != 0 and skip_read and not record_now:
            # 当不处于倒带和摄影模式时快速读取当前帧的数据但不解码，节省时间
            sample.grab()
            print('Skip', current_frame, 'frame.')
        else:
            ret, frame = sample.read()
            if frame is not None:

                # region process and inspect
                src_frame = frame.copy()
                frame = cv2.resize(frame, new_size)

                if current_frame == 0:
                    cut_box = get_boxes(frame)
                frame = cut_timestamp(cut_box=cut_box, vis=frame)

                temp, frame = get_diff(frame, old_frame)

                position = get_position(
                    frame=frame,
                    sizes=new_size
                )
                mark_motion(
                    new_size=new_size,
                    position=position,
                    frame=frame,
                    src_frame=src_frame
                )
                # endregion

                # region switch modes
                if position in [0, 1, 2, 3, 4, 5]:
                    if skip_read:
                        assert not record_now
                        old_frame = old_frame
                        # 如果画面出现变化且不处于倒带状态，则开始倒带（倒带2帧），正向状态切为倒带状态（skip_read=False）
                        if current_frame >= 2:
                            current_frame -= 2
                        else:
                            current_frame = 0
                        sample.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                        skip_read = False
                    else:
                        if record_now:
                            old_frame = temp
                            print('Recording...')
                            if vw is None:
                                vw = cv2.VideoWriter(
                                    output_path + "/" + file_name + "_" + str(file_count) + '.mp4',
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps,
                                    size
                                )
                            vw.write(frame)
                        else:
                            old_frame = old_frame
                            # 如果画面出现变化且处于倒带状态，则继续倒带（倒带2帧）
                            if current_frame >= 2:
                                current_frame -= 2
                            else:
                                current_frame = 0
                            sample.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                            skip_read = False
                else:
                    old_frame = temp
                    if skip_read:
                        # 如果画面无变化且不处于倒带状态，则继续运行，下一次将读取skip_frame个帧数后的那一帧
                        # 录像模式若开启，则关闭
                        assert not record_now
                    else:
                        if record_now:
                            record_now = False
                        if vw is not None:
                            if vw.isOpened():
                                vw.release()
                                file_count += 1
                        else:
                            # 如果画面无变化且处于倒带状态，则意味着到达有动作状态的开头，此时关闭倒带模式，开启录像模式
                            skip_read = False
                            record_now = True
                # endregion

    sample.release()
    if vw.isOpened:
        vw.release()
    cv2.destroyAllWindows()
    return src_id


def start_test_time(src_id, file_path):
    sample = cv2.VideoCapture(file_path)
    while sample.isOpened():
        ret, frame = sample.read()
        if frame is None:
            break
    sample.release()
    cv2.destroyAllWindows()
    return src_id
