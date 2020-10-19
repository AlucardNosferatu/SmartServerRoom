class ImrBaseException(Exception):
    def __init__(self,errcode,ErrorInfo):
        super().__init__(self) #初始化父类
        self.errorinfo=ErrorInfo
        self.errorcode = errcode
    def __str__(self):
        return  str(self.errorcode)+':'+self.errorinfo

if __name__ == '__main__':
    try:
        raise ImrBaseException(-1,'客户异常')
    except ImrBaseException as e:
        print(e)