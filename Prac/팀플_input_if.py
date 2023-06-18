import sys

A = input('팀 프로젝트 발표 자료 제출시간을 지켰나? : ')

if A in '지켰다':
    print('팀 프로젝트 발표')
elif A in '못지켰다':
    print('팀 프로젝트 발표 후 리젝')
    sys.exit()

B= input('팀 프로젝트 주제가 새롭고 유용한가?: ')

if B in '맞다':
    print('팀 프로젝트 진행')
elif B in '아니다':
    print('리젝')
    sys.exit()

C = input('팀 프로젝트가 3개월 정도의 난이도인가? : ')

if C in '맞다':
    print('억까를 당한 후 기능 추가 팀 프로젝트 진행')
elif C in '아니다':
    print('여러가지 이유로 혼난 후 리젝')
    sys.exit()

D = input('')