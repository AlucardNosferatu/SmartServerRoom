no_found = -1
# no_found = 'no such id'

test_img_path = "Samples/selfie.jpg"
predictor_path = 'Models/shape_predictor_68_face_landmarks.dat'
face_rc_model_path = 'Models/dlib_face_recognition_resnet_model_v1.dat'
face_folder_path = 'Backup/Faces'
# face_folder_path = 'C:/Users/16413/Documents/GitHub/YOLO/faces/Faces/forDlib'

ATOM_code = {
    'fd': '/imr-ai-service/atomic_functions/faces_detect',
    'ld': '/imr-ai-service/atomic_functions/landmarks_detect',
    'fr': '/imr-ai-service/atomic_functions/recognize',
    'rr': '/imr-ai-service/atomic_functions/reload',
    'ss': '/imr-ai-service/atomic_functions/snapshot',
}
# CEPH_code = {
#     'query': '/imr-ceph-server/ceph/query/',
#     'upload': '/imr-ceph-server/ceph/upload/',
#     'save': '/imr-ceph-server/ceph/save/'
# }
CEPH_code = {
    'query': '/ceph-server/ceph/query/',
    'upload': '/ceph-server/ceph/upload/',
    'save': '/ceph-server/ceph/save/'
}

# server_ip = 'http://134.134.13.81:29999'
# server_ip_2 = 'http://134.134.13.83:15656'
# server_ip_3 = 'http://134.134.13.84:15656'
server_ip = 'http://192.168.14.212:29999'
server_ip_2 = 'http://192.168.14.212:15656'
server_ip_3 = 'http://192.168.14.212:15656'

# callback_interface = {
#     'camera': 'http://134.134.13.82:8744/imr-face-server/faceapply_collection/faceCameraRecognResp'
# }
callback_interface = {
    'camera': 'http://192.168.14.212:8744/imr-face-server/faceapply_collection/faceCameraRecognResp'
}

save_path = 'Faces_Temp'
