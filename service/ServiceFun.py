import   autopower.AutoFun as  autop
'''原子能力第一个'''
def testService(param):
    res={}
    fi1= autop.autofun1()
    ffun2=autop.autofun2(param)
    res['function1']=fi1
    res['function2'] = ffun2
    return res
