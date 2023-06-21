import sys

A = input('개념공부 : ')


if A in '이해함':
    print('실습을 한다.')
elif A in '이해 못함':
    print('망함 ㅋㅋ')
    sys.exit()
    
B = input('실습을 한다. : ')

if B in '푼다':
    print('별거 아니네 ㅋㅋ')
elif B in '못푼다':
    print('답지가 있나?')
    C = input('여부 : ')
    if C in '있다':
        print('코드를 보고 감탄한다.')
    elif C in '없다':
        print('망함 ㅋㅋ')

