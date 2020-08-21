import os

import cv2
import zxing
import numpy as np
from tqdm import tqdm


def batch_filter():
    # path = 'D:/BaiduNetdiskDownload/202005/202005.v2'
    path = '/tmp/Photos'
    zx = zxing.BarCodeReader()
    img_list = os.listdir(path)
    img_list.sort()
    with open('valid_file.txt', mode='a+', encoding='utf-8') as f:
        f.seek(0)
        checked_length = len(f.readlines())
        f.seek(checked_length)
        for file_name in tqdm(img_list[checked_length:]):
            file_path = os.path.join(path, file_name)
            has_qr = single_image_test(zx, file_path, use_enhance=True)
            if has_qr:
                f.write(file_name + '\t' + '1' + '\n')
                f.flush()
            else:
                f.write(file_name + '\t' + '0' + '\n')
                f.flush()
    print('Done')


def single_image_test(zx, file_path, use_enhance):
    new_path = file_path
    image = cv2.imread(new_path)
    zx_data = zx.decode(new_path, try_harder=True)
    count = 0
    while zx_data is not None and len(zx_data.points) >= 3:
        if 'http://xfujian.189.cn' not in zx_data.raw:
            image = cv2.fillPoly(image, np.array([zx_data.points], dtype=np.int32), (255, 255, 255))
            new_path = new_path.replace('.jpg', str(count).join(['_', '.jpg']))
            cv2.imwrite(new_path, image)
            count += 1
            zx_data = zx.decode(new_path, try_harder=True)
            os.remove(new_path)
        else:
            return True
    if count == 0 and use_enhance:
        image_e = cv2.convertScaleAbs(image, alpha=1.8, beta=-90)
        cv2.imwrite(new_path, image_e)
        return single_image_test(zx, file_path, use_enhance=False)
    return False


if __name__ == '__main__':
    batch_filter()
    # zx = zxing.BarCodeReader()
    # file_path = 'Samples/temp.jpg'
    # single_image_test(zx, file_path, use_enhance=True)
