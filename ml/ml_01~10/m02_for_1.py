# for i in 리스트 데이터 : 

list  = ['a', 'b', 'c', 4]
#list형식에서는 형식 달라도 됨/ 그러나, numpy형태에서는 안됨 (같은 형식이어야 함)
for i in list:
    print(i)
'''
a
b
c
4
'''
for index, value in enumerate(list): # #enumerate : 순서대로 반환함 
    print(index, value)
    
'''
0 a
1 b
2 c
3 4
'''