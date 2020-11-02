import cv2
import time
import threading
from urllib import parse

from utils_FR import process_request, array2b64string


def post_t(b64string):
    result = process_request('fd_dbf', req_dict={'imgString': b64string})
    print(result)


def test_dbf():
    r = 'rtsp://admin:zww123456.@192.168.56.111:5542/h264/ch1/sub/av_stream'
    r = r.replace('+', parse.quote('+'))
    # output_name = 'Samples/fuck.mp4'
    # video_w = cv2.VideoWriter(
    #     output_name,
    #     cv2.VideoWriter_fourcc(*'mp4v'),
    #     25,
    #     (1024, 768)
    # )
    sample = cv2.VideoCapture(r)
    count = 0
    fla = True
    while fla:
        ret, fr = sample.read()
        if ret:
            if count % 5 == 0:
                time.sleep(2)
            fr = cv2.resize(fr, (1024, 768))
            b64string = array2b64string(fr).decode()
            t_post = threading.Thread(
                target=post_t, args=(
                    b64string,
                )
            )
            t_post.start()
            # cv2.imshow('rtsp', fr)
            # video_w.write(fr)
            count += 1
            # cv2.waitKey(1)
        # if count > 500:
        #     break
    # video_w.release()
    sample.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_dbf()
