import pygame
import pandas as pd
import ast
pygame.init()

dis = pygame.display.set_mode((500, 500))

clock = pygame.time.Clock()
direction = 'Right'

game_data=pd.read_csv("./Snakes/Best Snake gen49.csv")
# game_data=pd.read_csv("Random snake.csv")
red = (255,0,0)
green=(0,255,0)
window_size=400
block_size=window_size/10  #divided by the number of squares in game




dis = pygame.display.set_mode((window_size, window_size+200))
font = pygame.font.Font(pygame.font.get_default_font(), 16)
run = True




def info_below(data):
    left,right,front,applex,appley,snakex,snakey,score=data
    score_surface = font.render(str(score),True,red)
    dis.blit(score_surface, dest=(200,0))

    pygame.draw.rect(dis, red, pygame.Rect(185, 485, block_size, block_size))
    pygame.draw.rect(dis, green, pygame.Rect(50, 485, block_size, block_size))

    applex_surface = font.render(str(applex),True,red)
    dis.blit(applex_surface, dest=(10,500))

    appley_surface = font.render(str(appley),True,red)
    dis.blit(appley_surface, dest=(75,470))


    left_surface = font.render(str(left),True,red)
    dis.blit(left_surface, dest=(150,400+100))

    right_surface = font.render(str(right),True,red)
    dis.blit(right_surface, dest=(250,400+100))

    front_surface = font.render(str(front),True,red)
    dis.blit(front_surface, dest=(200,400+50))
    
    



def animate(frame):
    row=game_data.iloc[frame,:]
    if row["run"]=="penalty" or row["run"]=="death":
        return None
    
    info_below(row.iloc[5:].tolist())
    

    snake_pos=ast.literal_eval(row["Pos. snake"])
    apple_pos=ast.literal_eval(row["Pos. apple"])
    for point in snake_pos:
        pygame.draw.rect(dis, red, pygame.Rect(point[0]*block_size, point[1]*block_size, block_size, block_size))
    pygame.draw.rect(dis, green, pygame.Rect(apple_pos[0]*block_size, apple_pos[1]*block_size, block_size, block_size))

frame=0
while run:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    keys = pygame.key.get_pressed()
    dis.fill((0, 0, 255),(0, window_size, window_size, 200))
    dis.fill((0, 0, 0),(0, 0, window_size, window_size))
    animate(frame)
    frame+=1
    pygame.display.flip()
    clock.tick(20)

pygame.quit()