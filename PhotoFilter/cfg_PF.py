import os

work_dir = __file__
if '\\' in work_dir:
    work_dir = work_dir.split('\\')
    work_dir = work_dir[:-1]
    work_dir = '\\'.join(work_dir)
else:
    work_dir = work_dir.split('/')
    work_dir = work_dir[:-1]
    work_dir = '/'.join(work_dir)
# no_found = 'no such id'
no_found = -1

CEPH_code = {
    'query': '/ceph-server/ceph/query/',
    'upload': '/ceph-server/ceph/upload/',
    'save': '/ceph-server/ceph/save/'
}
# CEPH_code = {
#     'query': '/imr-ceph-server/ceph/query/',
#     'upload': '/imr-ceph-server/ceph/upload/',
#     'save': '/imr-ceph-server/ceph/save/'
# }
ATOM_code = {
    'qr': '/imr-ai-service/atomic_functions/qrcode_decode',
    'dl': '/imr-ai-service/atomic_functions/sticker_detect',
}

# api_server = 'http://192.168.14.212:29999'
api_server = 'http://134.134.13.81:29999'

# download_server = 'http://192.168.254.169'
download_server = 'http://134.134.13.152:8888'

qrc_save_path = os.path.join(work_dir, 'QRC_Temp')
