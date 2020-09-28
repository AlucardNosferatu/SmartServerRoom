from flask import current_app
'''打印日志'''
def info(log):
    current_app.logger.warning(log)

def error(log):
    current_app.logger.error(log)