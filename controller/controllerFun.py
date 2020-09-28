from flask import Blueprint
import  service.ServiceFun as  servicefun
import   util.ResultUtil  as   resultutil
import logger.logger as logger
'''注册蓝图'''
controllerFun = Blueprint('simple',__name__,template_folder='templates')
'''
测试函数
'''
@controllerFun.route('/testFun/<filename>', methods=['POST'])
def wadadwad(filename):
    logger.info("filename"+filename)
    res  =  servicefun.testService(filename)
    return resultutil.success(res)