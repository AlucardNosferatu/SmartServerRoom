import base64

from utils import process_request

path = 'Samples/cropped_20200927053934149_ff633cbe.png'
with open(path, 'rb') as fc:
    b64_string = base64.b64encode(fc.read())
    b64_string = b64_string.decode()
    b64_string = 'data:image/jpeg;base64,' + b64_string
points = process_request('ld', req_dict={'imgString': b64_string})