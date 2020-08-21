import datetime
import os
import time

import cv2
import numpy as np
from avtk.backends.ffmpeg.convert import FFmpeg, Output, Video, NoAudio

from Analysis import start_test_lite, start_test_time, start_test_new
from HTTPInterface import post_result, MyRequestHandler, HTTPServer
from MostDifferentFrame import snap_shot


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


def process_dir(
        _,
        request_id,
        dir_path="C:\\Users\\16413\\Desktop\\FFCS\\SVN\\CV_Toolbox\\SmartServerRoom\\Samples",
        output_path="C:\\Users\\16413\\Desktop\\FFCS\\SVN\\CV_Toolbox\\SmartServerRoom\\Outputs"
):
    indices = range(245, 289)
    if type(dir_path) == list:
        dir_path = dir_path[0]
    if type(output_path) == list:
        output_path = output_path[0]
    src_num = 0
    dst_num = 0
    src_id = 0
    busy = True
    in_files = []
    while busy:
        try:
            in_files = os.listdir(dir_path)
        except OSError:
            continue
        busy = False
    for e, i in enumerate(in_files):
        if (i.endswith('mp4') or i.endswith('MP4')) and True:
            file_path = os.path.join(dir_path, i)
            print('file_path is:', file_path)
            start = datetime.datetime.now()
            retry = 5
            while retry > 0:
                try:
                    convert(file_path=file_path)
                    flv_path = file_path.replace('.mp4', '.MP4').replace('.MP4', '.flv')
                    src_id = start_test_lite(
                        src_id=request_id,
                        file_path=flv_path,
                        output_path=output_path,
                        file_name=i,
                        skip_read=False,
                        show_diff=False
                    )
                    # src_id = start_test_new(
                    #     src_id=request_id,
                    #     file_path=flv_path,
                    #     output_path=output_path,
                    #     file_name=i
                    # )
                    if os.path.exists(flv_path):
                        os.remove(flv_path)
                    retry = 0
                except Exception as e:
                    print(repr(e))
                    retry -= 1
                    time.sleep(1)
                    continue
            end = datetime.datetime.now()
            print(str(end - start))
            src_num += 1
    busy = True
    out_files = []
    while busy:
        try:
            out_files = os.listdir(output_path)
        except OSError:
            continue
        busy = False
    for e, i in enumerate(out_files):
        if i.endswith('mp4') and True:
            file_path = os.path.join(output_path, i)
            retry = 5
            while retry > 0:
                try:
                    snap_shot(calc_and_draw_hist, file_path=file_path)
                    retry = 0
                except Exception as e:
                    print(repr(e))
                    retry -= 1
                    time.sleep(1)
                    continue
            dst_num += 1
    assert src_id == request_id
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


def convert(file_path='Samples/010tMonitorCollect20200729025635377826769671_171.mp4'):
    output_path = file_path.replace('.mp4', '.MP4').replace('.MP4', '.flv')
    start = datetime.datetime.now()
    FFmpeg(
        file_path,
        Output(
            output_path,
            streams=[
                Video('flv', scale=(-1, 768), bit_rate='1500k'),
                NoAudio
            ],

        )
    ).run()
    # convert_to_h264(file_path, 'Outputs/test.mp4', preset='fast')
    end = datetime.datetime.now()
    print(str(end - start))


if __name__ == '__main__':
    start_server()
    # process_dir(_=None, request_id='1')
