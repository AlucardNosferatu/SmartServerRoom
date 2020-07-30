import os

import zxing
from tqdm import tqdm


def batch_filter():
    # path = 'C:/BaiduNetdiskDownload/202005/202005.v2'
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
            zx_data = zx.decode(file_path)
            if zx_data is not None and len(zx_data.raw) > 0:
                f.write(file_name + '\t' + '1' + '\n')
                f.flush()
            else:
                f.write(file_name + '\t' + '0' + '\n')
                f.flush()

    print('Done')


if __name__ == '__main__':
    batch_filter()
