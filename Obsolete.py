import datetime
from math import inf

import cv2
import numpy as np

from Analysis import get_diff, get_position, mark_motion, trigger
from CamMonitor import calc_and_draw_hist, convert
from ShapeFilter import valid_shape
from TimeStamp import get_boxes, cut_timestamp


def start_test_new(
        src_id,
        file_path="Samples\\Sample.mp4",
        output_path="Outputs",
        file_name='Sample.mp4',
        skip_frame=100
):
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
                cv2.imshow('cutts', frame)
                cv2.waitKey()
                if current_frame == 0:
                    cut_box = get_boxes(frame)
                frame = cut_timestamp(cut_box=cut_box, vis=frame)

                old_frame, frame = get_diff(frame, old_frame)
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
                        # 如果画面出现变化且不处于倒带状态，则开始倒带（倒带2帧），正向状态切为倒带状态（skip_read=False）
                        if current_frame >= 2:
                            current_frame -= 2
                        else:
                            current_frame = 0
                        sample.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                        skip_read = False
                    else:
                        # 如果画面出现变化且处于倒带状态，则继续倒带（倒带2帧）
                        if current_frame >= 2:
                            current_frame -= 2
                        else:
                            current_frame = 0
                        sample.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                        skip_read = False
                else:
                    if skip_read:
                        # 如果画面无变化且不处于倒带状态，则继续运行，下一次将读取skip_frame个帧数后的那一帧
                        # 录像模式若开启，则关闭
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

                if record_now:
                    print('Recording...')
                    # if vw is None:
                    #     vw = cv2.VideoWriter(
                    #         output_path + "/" + file_name + "_" + str(file_count) + '.mp4',
                    #         cv2.VideoWriter_fourcc(*'mp4v'),
                    #         fps,
                    #         size
                    #     )
                    # vw.write(frame)

    sample.release()
    if vw.isOpened:
        vw.release()
    cv2.destroyAllWindows()
    return src_id


def start_test(
        src_id,
        # skip_frame=7,
        show_diff=False,
        file_path="Samples\\Sample.mp4",
        output_path="Outputs",
        file_name="Sample.mp4",
        skip_read=False
):
    start_time = datetime.datetime.now()
    file_name = file_name.split(".")[0]
    print(file_path)
    print(output_path)

    # region Initialize variables
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
    # record = []
    th = 1000
    use_diff = True
    file_count = 0
    next_move = 200
    first_frame = True
    cut_box = [[57, 25, 500]]
    # endregion

    # region Initialize VideoWriter
    fps = sample.get(cv2.CAP_PROP_FPS)
    br = sample.get(cv2.CAP_PROP_BITRATE)
    skip_frame = int(5 * int(br / 1000)) - 5
    # skip_frame = 5
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
    base_time = datetime.datetime.now()
    fr = base_time
    gr = base_time
    pvb = base_time
    ct = base_time
    diff_t = base_time
    flicker = base_time
    histo_t = base_time
    trigger_t = base_time
    record_t = base_time
    mm_t = base_time
    count = 0
    while sample.isOpened():
        now = datetime.datetime.now()
        if count % (skip_frame + 1) != 0 and skip_read:
            count += 1
            sample.grab()
            next_move -= 1
            if next_move < 0:
                next_move = 0
                if video_writer.isOpened():
                    video_writer.release()
            then = datetime.datetime.now()
            print('frame grab: ', str(then - now))
            gr += (then - now)
            continue

        ret, frame = sample.read()
        then = datetime.datetime.now()
        print('frame read: ', str(then - now))
        fr += (then - now)
        now = then

        if frame is not None:
            # print(next_move)
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
                then = datetime.datetime.now()
                print("Start recording: ", str(then - now))
                now = then
            # frame = enhance(frame)

            # region get Differential Frame
            # prev_frames = []
            src_frame = frame.copy()
            prev_frames.append(src_frame)

            if len(prev_frames) > 20:
                prev_frames.pop(0)
            then = datetime.datetime.now()
            print('Push video buffer: ', str(then - now))
            pvb += (then - now)
            now = then

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
                then = datetime.datetime.now()
                print('Record during skipped frame: ', str(then - now))
                record_t += (then - now)
                now = then
                continue

            frame = cv2.resize(frame, (1024, 768))
            if first_frame:
                cut_box = get_boxes(frame)
                then = datetime.datetime.now()
                print('Get timestamp from first frame: ', str(then - now))
                ct += (then - now)
                now = then

            frame = cut_timestamp(cut_box=cut_box, vis=frame)
            then = datetime.datetime.now()
            print('Cut timestamp: ', str(then - now))
            ct += (then - now)
            now = then

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
            then = datetime.datetime.now()
            print('Get diff of frames: ', str(then - now))
            diff_t += (then - now)
            now = then

            # region get Sizes
            h = frame.shape[0]
            w = frame.shape[1]
            x1 = int(0.3 * w)
            x2 = int(0.7 * w)
            y = int(0.5 * h)
            # endregion

            flicker_points = valid_shape(frame)
            then = datetime.datetime.now()
            print('Detect flicker: ', str(then - now))
            flicker += (then - now)
            now = then
            # if flicker_points:
            #     print("FP")

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
            then = datetime.datetime.now()
            print("get histo: ", str(then - now))
            histo_t += (then - now)
            now = then

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
            p, sig = trigger(hist_list, th)
            old1, old2, old3, old4, old5, old6 = hist_list
            position = p
            then = datetime.datetime.now()
            print('Get trigger: ', str(then - now))
            trigger_t += (then - now)
            now = then
            # if sig > 2 * th:
            #     sig = 2 * th
            # record.append(sig)
            # if len(record) > 200:
            #     del record[0]

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
                # print("Recording")
                video_writer.write(prev_frames[0])
                then = datetime.datetime.now()
                print('Recording: ', str(then - now))
                record_t += (then - now)
                now = then

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
                then = datetime.datetime.now()
                print("Mark motion: ", str(then - now))
                mm_t += (then - now)
                now = then
            # endregion
            if show_diff:
                cv2.imshow("diff", frame)
                cv2.imshow("origin", cv2.resize(src_frame, (1024, 768)))
                # cv2.imshow("cut_ts", inspect_frame)
                # cv2.waitKey()
        else:
            break
        first_frame = False
        # region key interface
        k = cv2.waitKey(1)
        if k & 0xff == ord('q'):
            print("redefine threshold")
            th = th * 1.05
        if k & 0xff == ord('e'):
            print("redefine threshold")
            th = th * 0.95
        if k & 0xff == ord('w'):
            break
        # endregion
    fr -= base_time
    gr -= base_time
    pvb -= base_time
    ct -= base_time
    diff_t -= base_time
    flicker -= base_time
    histo_t -= base_time
    trigger_t -= base_time
    record_t -= base_time
    mm_t -= base_time
    print('total_fr: ', str(fr))
    print('total_gr: ', str(gr))
    print('total_pvb: ', str(pvb))
    print('total_ct: ', str(ct))
    print('total_diff: ', str(diff_t))
    print('total_flicker: ', str(flicker))
    print('total_histo: ', str(histo_t))
    print('total_trigger: ', str(trigger_t))
    print('total_record: ', str(record_t))
    print('total_motion:', str(mm_t))
    sample.release()
    if video_writer.isOpened:
        video_writer.release()
    cv2.destroyAllWindows()
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print('Duration: ', str(duration))
    return src_id


if __name__ == '__main__':
    convert()
