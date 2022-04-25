import pygame
from pygame.locals import *
import time
import random
import pandas as pd
import time
SIZE = 40
BACKGROUND_COLOR = (110, 110, 5)
game_data=pd.read_csv("Random snake.csv")
red = (255,0,0)
green=(0,255,0)
game_size=100
import ast
surface=pygame.display.set_mode((game_size*8, game_size*8))


def game_step(row):
    snake_pos=ast.literal_eval(row["Pos. snake"])
    apple_pos=ast.literal_eval(row["Pos. apple"])
    for point in snake_pos:
        pygame.draw.rect(surface, red, pygame.Rect(point[0]*8, point[1]*8, 8, 8))
    pygame.draw.rect(surface, green, pygame.Rect(apple_pos[0]*8, apple_pos[1]*8, 8, 8))
    pygame.display.flip()
    surface.fill((0,0,0))
    time.sleep(2)






game_data.apply(lambda x: game_step(x),axis=1)

