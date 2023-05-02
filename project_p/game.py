import pygame
import random

# 색깔 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# 화면 크기 설정
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400

# 지렁이 클래스
class Worm:
    # 초기화 함수
    def __init__(self, x, y):
        self.segments = [(x, y), (x-10, y), (x-20, y)]
        self.dx = 10
        self.dy = 0
    
    # 이동 함수
    def move(self):
        # 꼬리를 자르고 머리에 새로운 세그먼트 추가
        self.segments = [(self.segments[0][0]+self.dx, self.segments[0][1]+self.dy)] + self.segments[:-1]
    
    # 방향 설정 함수
    def set_direction(self, dx, dy):
        self.dx = dx
        self.dy = dy
    
    # 지렁이 그리기 함수
    def draw(self, surface):
        for segment in self.segments:
            pygame.draw.rect(surface, WHITE, (segment[0], segment[1], 10, 10))

# 먹이 클래스
class Food:
    # 초기화 함수
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH-10)
        self.y = random.randint(0, SCREEN_HEIGHT-10)
    
    # 먹이 그리기 함수
    def draw(self, surface):
        pygame.draw.rect(surface, RED, (self.x, self.y, 10, 10))

    # 위치 변경 함수
    def move(self):
        self.x = random.randint(0, SCREEN_WIDTH-10)
        self.y = random.randint(0, SCREEN_HEIGHT-10)

# 게임 함수
def game():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # 객체 생성
    worm = Worm(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
    food = Food()
    
    # 게임 루프
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    worm.set_direction(-10, 0)
                elif event.key == pygame.K_RIGHT:
                    worm.set_direction(10, 0)
                elif event.key == pygame.K_UP:
                    worm.set_direction(0, -10)
                elif event.key == pygame.K_DOWN:
                    worm.set_direction(0, 10)
        
        # 지렁이 이동
        worm.move()
        
        # 충돌 검사
        if worm.segments[0][0] < 0 or worm.segments[0][0] >= SCREEN_WIDTH or worm.segments[0][1] < 0 or worm.segments[0][1] >= SCREEN_HEIGHT:
            pygame.quit()
            quit()
        elif worm.segments[0] == (food.x, food.y):
            food.move()
            worm.segments.append(worm.segments[-1])
        
        # 화면 그리기
        screen.fill(BLACK)
        worm.draw(screen)
        food.draw(screen)
        pygame.display.update()
        
        # 프레임 설정
        clock.tick(10)

# 게임 실행
game()
