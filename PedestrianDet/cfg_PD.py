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

ATOM_code = {
    'pd': '/imr-ai-service/atomic_functions/detect_pedestrian'
}

query_temp_api = '/ceph-server/ceph/querytemp/'
query_api = '/ceph-server/ceph/query/'
upload_api = '/ceph-server/ceph/upload/'
save_api = '/ceph-server/ceph/save/'

# callback_interface = {
#     'ped': 'http://192.168.14.212:8744/imr-face-server/prepareimage/checkperson/',
#     'listener': 'http://127.0.0.1:20295/imr-face-server/callback_listener'
# }
callback_interface = {
    'ped': 'http://134.134.13.82:8744/imr-face-server/prepareimage/checkperson/',
    'listener': 'http://127.0.0.1:20295/imr-face-server/callback_listener'
}

# api_server = 'http://192.168.14.212:29999'
api_server = 'http://134.134.13.81:29999'

# download_server = 'http://192.168.254.169'
download_server = 'http://134.134.13.152:8888'

save_path = os.path.join(work_dir, 'P_Temp')
