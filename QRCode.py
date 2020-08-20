import os

import cv2
import zxing
import numpy as np
from tqdm import tqdm


def batch_filter():
    path = 'D:/BaiduNetdiskDownload/202005/202005.v2'
    # path = '/tmp/Photos'
    zx = zxing.BarCodeReader()
    img_list = os.listdir(path)
    img_list.sort()
    with open('valid_file.txt', mode='a+', encoding='utf-8') as f:
        f.seek(0)
        checked_length = len(f.readlines())
        f.seek(checked_length)
        for file_name in tqdm(img_list[checked_length:]):
            file_path = os.path.join(path, file_name)
            has_qr = single_image_test(zx, file_path)
            if has_qr:
                f.write(file_name + '\t' + '1' + '\n')
                f.flush()
            else:
                f.write(file_name + '\t' + '0' + '\n')
                f.flush()
    print('Done')


def single_image_test(zx, file_path):
    new_path = file_path
    zx_data = zx.decode(new_path)
    count = 0
    while zx_data is not None and len(zx_data.raw) > 0:
        if 'http://xfujian.189.cn' not in zx_data.raw:
            image = cv2.imread(new_path)
            image = cv2.fillPoly(image, np.array([zx_data.points], dtype=np.int32), (255, 255, 255))
            new_path = new_path.replace('.jpg', str(count).join(['_', '.jpg']))
            cv2.imwrite(new_path, image)
            count += 1
            zx_data = zx.decode(new_path)
        else:
            return True
    return False


if __name__ == '__main__':
    batch_filter()
