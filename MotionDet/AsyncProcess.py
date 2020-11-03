import os

from cfg_MD import save_path
from utils_MD import download, process_request, upload, response_async


def convert_async(file_id, trance_log_id):
    file_name = download(req_id=file_id, from_temp=False)
    result = file_name
    if result != -1:
        params = {
            'file_path': os.path.join(save_path, file_name),
            'codec': 'libx264',
            'postfix': '.flv',
            'bitRate': '1500k',
            'scale': None,
            'deletion': True
        }
        result = process_request('vc', params)
    if os.path.exists(os.path.join(save_path, file_name)):
        os.remove(os.path.join(save_path, file_name))
    if result['res'] != -1:
        result = upload(file_name=result['res'], to_temp=False, deletion=True, file_dir='')
    print(result)
    if trance_log_id != 'LOCAL_USAGE':
        response_async(result, 'convert', url_param=trance_log_id)
        result_with_id = result.copy()
        result_with_id['trance_log_id'] = trance_log_id
        response_async(result=result_with_id, function='listener')
    return result
