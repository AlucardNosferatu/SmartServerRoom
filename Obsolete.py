import cv2

from Analysis import get_diff, get_position, mark_motion
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
                cv2.imshow('cutts',frame)
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