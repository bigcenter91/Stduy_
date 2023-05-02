# for문은 list 형태로 나옴.
# for i in 리스트 데이터 : 

list = ['a','b','c', 4] #4가 들어가도 가능. 문자와 숫자가 섞어서도 가능.

for i in list :
    print(i)

# a
# b
# c

for index, value in enumerate(list):
    print(index, value)
    
# 0 a
# 1 b
# 2 c   -> 순서도 같이 빼고싶으면 enumerate를 사용.

#리스트는 이터레이터에 포함된다. for문이면 이터레이터에 들어갈 수 있음.

