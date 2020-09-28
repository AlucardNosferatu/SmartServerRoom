from flask import Flask

app = Flask('ellisTest')

'''自己测蓝图'''
from controller.controllerFun import controllerFun
app.register_blueprint(controllerFun,url_prefix='/controllerFun')




app.run(
    host="0.0.0.0",
    port=int("7120"),
    debug=False, threaded=True)

