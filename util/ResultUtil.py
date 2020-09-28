def  success(data):
    result = {}
    result['data'] = data
    result['code'] = 0
    result['ret']  =True
    result['msg'] ='成功'
    return result

def  failed(errcode,msg):
    result = {}
    result['data'] = None;
    result['errcode'] = errcode
    result['ret']  =True
    result['msg'] =msg
    return result