class Test:
    abc = 10  # public
    _abc = 30  # public 
    __abc = 20 # private

    def __init__(self):
        self.hhh=50

    def add(self,x,y):  #public
        #return self.abc+ x  #可以访问public变量
        #return self.__abc + x #可以访问 private 变量
        self.__t()
        
        return self.ttt +x   # 不可以访问
        return self.vtt+x
        return x+y
    def __add2(self,x,y):  #private
        return x+y
    
    def __t(self):
        self.ttt=10
        print(self.ttt)
    def tt(self):
        self.vtt=60
        print(self.vtt)

t=Test()

t.tt()

print(t.abc,t._abc,t.add(8,9))