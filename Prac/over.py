class father(): #부모 클래스
    def handsome(self):
        print("잘생겼다")
 
class brother(father): #자식클래스(부모클래스) 아빠매소드를 상속받겠다
    '''아들'''
 
class sister(father): #자식클래스(부모클래스) 아빠매소드를 상속받겠다
    def pretty(self):
        print("예쁘다")
    def handsome(self):
        self.pretty()
 
brother = brother()
brother.handsome()
 
girl = sister()
girl.handsome()
