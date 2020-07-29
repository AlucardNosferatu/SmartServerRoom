import os

import cv2
import zxing
from tqdm import tqdm


def batch_filter():
    pass_list = []
    path = 'C:/BaiduNetdiskDownload/202005/202005.v2'
    zx = zxing.BarCodeReader()
    img_list = os.listdir(path)
    for file_name in tqdm(img_list):
        file_path = os.path.join(path, file_name)
        zx_data = zx.decode(file_path)
        if len(zx_data.raw) > 0:
            # img_array = cv2.imread(file_path)
            # cv2.imshow('detected',img_array)
            # cv2.waitKey()
            pass_list.append(file_name + '\n')
    with open('valid_file.txt', mode='w', encoding='utf-8') as f:
        f.writelines(pass_list)

    print('Done')


if __name__ == '__main__':
    batch_filter()
