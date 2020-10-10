
from flask import Flask

import util.ResultUtil as resututl
from exception.ImrBaseException import ImrBaseException

app = Flask('ImrProject')

'''自己测蓝图'''
from controller.VideoController import videocontroller
app.register_blueprint(videocontroller,url_prefix='/videocontroller')


@app.errorhandler(Exception)
def all_exception_handler(e):
    # 对于 HTTP 异常，返回自带的错误描述和状态码
    # 这些异常类在 Werkzeug 中定义，均继承 HTTPException 类
    if isinstance(e, ImrBaseException):
        return resututl.failed(e.errorcode,e.errorinfo)
    return resututl.failed(-12,str(e))

app.run(
    host="0.0.0.0",
    port=int("7120"),
    debug=False, threaded=True)

