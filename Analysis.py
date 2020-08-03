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
        cv2.imshow("diff", cv2.resize(frame, (512, 384)))
        cv2.imshow("origin", cv2.resize(src_frame, (512, 384)))
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


def mode_switch(position, current_mode, wait_rewind, wait_record):
    if position in [0, 1, 2, 3, 4, 5]:
        if current_mode == "fast_forward":
            return "rewind", wait_rewind, wait_record
        elif current_mode == "rewind":
            wait_rewind = 5
            return "rewind", wait_rewind, wait_record
        elif current_mode == "start_record":
            return "recording", wait_rewind, wait_record
        elif current_mode == "recording":
            wait_record = 100
            return "recording", wait_rewind, wait_record
        elif current_mode == "stop_record":
            return "start_record", wait_rewind, wait_record
        else:
            raise ValueError("状态异常")
    else:
        if current_mode == "fast_forward":
            return "fast_forward", wait_rewind, wait_record
        elif current_mode == "rewind":
            wait_rewind -= 1
            if wait_rewind <= 0:
                wait_rewind = 5
                return "start_record", wait_rewind, wait_record
            else:
                return "rewind", wait_rewind, wait_record
        elif current_mode == "start_record":
            wait_record -= 1
            if wait_record <= 0:
                wait_record = 100
                return "stop_record", wait_rewind, wait_record
            else:
                return "recording", wait_rewind, wait_record
        elif current_mode == "recording":
            wait_record -= 1
            if wait_record <= 0:
                wait_record = 100
                return "stop_record", wait_rewind, wait_record
            else:
                return "recording", wait_rewind, wait_record
        elif current_mode == "stop_record":
            return "fast_forward", wait_rewind, wait_record
        else:
            raise ValueError("状态异常")


def fast_forward(sample, current_time, skip_frame):
    # 当不处于倒带和摄影模式时快速读取当前帧的数据但不解码，节省时间
    if current_time % (skip_frame * 80) != 0:
        sample.grab()
        return "skip_diff"
    else:
        ret, frame = sample.read()
        if frame is not None:
            return frame
        else:
            return "end_of_stream"


def rewind(current_time, sample):
    if current_time >= (80 * 20):
        current_time -= (80 * 20)
    else:
        current_time = 0

    sample.set(cv2.CAP_PROP_POS_MSEC, current_time)
    ret, frame = sample.read()
    if frame is None:
        frame = "end_of_stream"
    return sample, current_time, frame


def start_record(frame, output_path, file_name, file_count, fps, size, vw=None):
    if vw is None or not vw.isOpened():
        vw = cv2.VideoWriter(
            output_path + "/" + file_name + "_" + str(file_count) + '.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            size
        )
        file_count += 1
    vw.write(frame)
    return vw, file_count


def recording(vw, frame):
    assert vw.isOpened()
    vw.write(frame)


def stop_record(vw, frame):
    assert vw.isOpened
    vw.write(frame)
    vw.release()
    vw = None
    return vw


def process_and_inspect(frame, new_size, first_frame, old_frame, cut_box):
    # region process and inspect
    src_frame = frame.copy()
    frame = cv2.resize(frame, new_size)

    if first_frame:
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
    return position, src_frame, temp, cut_box


def start_test_new(
        src_id,
        file_path="Samples\\Sample.mp4",
        output_path="Outputs",
        file_name='Sample.mp4',
        skip_frame=6
):
    file_name = file_name.split(".")[0]
    cut_box = [[57, 25, 500]]
    new_size = (1024, 768)
    sample = cv2.VideoCapture(file_path)

    vw = None
    fps = int(sample.get(cv2.CAP_PROP_FPS) / 2)
    size = (
        int(sample.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(sample.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    temp = None
    file_count = 0
    position = -1
    wait_rewind = 5
    wait_record = 100
    current_mode = "fast_forward"
    first_frame = True
    while sample.isOpened():
        # current_frame = int(sample.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = int(sample.get(cv2.CAP_PROP_POS_MSEC))
        current_mode, wait_rewind, wait_record = mode_switch(
            position=position,
            current_mode=current_mode,
            wait_rewind=wait_rewind,
            wait_record=wait_record
        )
        print(current_mode, current_time, wait_rewind, wait_record, file_count)
        if current_mode == "fast_forward":
            frame = fast_forward(sample=sample, current_time=current_time, skip_frame=skip_frame)
            if type(frame) is str:
                if frame == "skip_diff":
                    continue
                else:
                    break
        elif current_mode == "rewind":
            sample, current_frame, frame = rewind(current_time=current_time, sample=sample)
            if type(frame) is str:
                break
        elif current_mode in ['start_record', 'recording', 'stop_record']:
            ret, frame = sample.read()
            sample.grab()
            if frame is None:
                break
        else:
            raise ValueError('状态异常')

        assert frame.shape == (size[1], size[0], 3)

        old_frame = temp

        position, src_frame, temp, cut_box = process_and_inspect(
            frame=frame,
            new_size=new_size,
            first_frame=first_frame,
            old_frame=old_frame,
            cut_box=cut_box
        )

        assert temp.shape == (new_size[1], new_size[0])
        assert src_frame.shape == (size[1], size[0], 3)

        if current_mode == "start_record":
            vw, file_count = start_record(
                frame=src_frame,
                output_path=output_path,
                file_name=file_name,
                file_count=file_count,
                fps=fps,
                size=size,
                vw=vw
            )
        elif current_mode == "recording":
            recording(vw=vw, frame=src_frame)
        elif current_mode == "stop_record":
            stop_record(vw=vw, frame=src_frame)

        first_frame = False
    sample.release()
    if vw is not None and vw.isOpened:
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
