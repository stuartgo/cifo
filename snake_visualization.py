import pygame
from pygame.locals import *
import time
import random
import pandas as pd
import time



""
#game_data=pd.read_csv("Random snake.csv")
game_data=pd.read_csv("./Snakes/Best Snake gen8.csv")
red = (255,0,0)
green=(0,255,0)
window_size=400
block_size=window_size/10  #divided by the number of squares in game
import ast
surface=pygame.display.set_mode((window_size, window_size))


def game_step(row):
    snake_pos=ast.literal_eval(row["Pos. snake"])
    apple_pos=ast.literal_eval(row["Pos. apple"])
    for point in snake_pos:
        pygame.draw.rect(surface, red, pygame.Rect(point[0]*block_size, point[1]*block_size, block_size, block_size))
    pygame.draw.rect(surface, green, pygame.Rect(apple_pos[0]*block_size, apple_pos[1]*block_size, block_size, block_size))
    pygame.display.flip()
    surface.fill((0,0,0))
    time.sleep(0.5)






game_data.apply(lambda x: game_step(x),axis=1)

