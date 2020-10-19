ATOM_code = {
    'hz': '/imr-ai-service/atomic_functions/hot_zone',
    '6h': '/imr-ai-service/atomic_functions/histograms_3x2',
    'di': '/imr-ai-service/atomic_functions/difference_between_frames',
    'ss': '/imr-ai-service/atomic_functions/most_different_frames',
    'cb': '/imr-ai-service/atomic_functions/cut_a_box',
    'ts': '/imr-ai-service/atomic_functions/timestamp',
    'vc': '/imr-ai-service/atomic_functions/convert'
}
# CEPH_code = {
#     'query': '/imr-ceph-server/ceph/query/',
#     'upload': '/imr-ceph-server/ceph/upload/',
#     'save': '/imr-ceph-server/ceph/save/'
# }
CEPH_code = {
    'query': '/ceph-server/ceph/query/',
    'qt': '/ceph-server/ceph/querytemp/',
    'upload': '/ceph-server/ceph/upload/',
    'save': '/ceph-server/ceph/save/'
}

server_ip = 'http://134.134.13.81:29999'
server_ip_2 = 'http://134.134.13.83:15656'
server_ip_3 = 'http://134.134.13.84:15656'
# server_ip = 'http://192.168.14.212:29999'
# server_ip_2 = 'http://192.168.14.212:15656'
# server_ip_3 = 'http://192.168.14.212:15656'

download_server = 'http://134.134.13.152:8888'
# download_server = 'http://192.168.254.169'

callback_interface = {
    'camera': 'http://134.134.13.82:8744/imr-face-server/faceapply_collection/faceCameraRecognResp/',
    'motion': 'http://134.134.13.82:8744/imr-face-server/monitor/regmonitor/',
    'convert': 'http://134.134.13.82:8744/imr-face-server/prepareimage/trance_read/'
}
# callback_interface = {
#     'camera': 'http://192.168.14.212:8744/imr-face-server/faceapply_collection/faceCameraRecognResp/',
#     'motion': 'http://192.168.14.212:8744/imr-face-server/monitor/regmonitor/',
#     'convert': 'http://192.168.14.212:8744/imr-face-server/prepareimage/trance_read/'
# }

save_path = 'M_Temp'
