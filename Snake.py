import pygame

from GAagent import GAAgent
import numpy as np
#import utils
import keyboard
from Game import Game
from Food import Food
from Player import Player

#this file is copied from https://github.com/davide97l/Snake-Battle-Royale/tree/master/snake
#This file can be used for visualizations


def run_snake():
    pygame.init()
    pygame.font.init()
    game = Game(20, 20)


    snake_red = Player(game, "red")
    game.player=snake_red


    game.food.append(Food(game))


    ga_agent = GAAgent(population_name="geom_seman", generation=49)
    snake_red.set_agent(ga_agent)

    game.game_speed = 100  # parameter: game speed
    game.display_option = True  # parameter: show game
    record = False  # parameter: True if recording the game
    frames = []
    run=True
    while not keyboard.is_pressed('s') and run:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        move = game.player.move(game)
        game.player.do_move(move, game)
        if game.player.crash:
            game.player.init_player(game)
        if game.display_option:
            game.display()
            pygame.time.wait(game.game_speed)
            # if record:
            #     data = pygame.image.tostring(game.gameDisplay, 'RGBA')
            #     from PIL import Image
            #     img = Image.frombytes('RGBA', (game.game_width, game.game_height + 100), data)
            #     img = img.convert('RGB')
            #     frames.append(np.array(img))

    print(" Max score: " + str(snake_red.record) +
            ", Avg Score: " + str(snake_red.total_score / snake_red.deaths) +
            ", Deaths: " + str(snake_red.deaths))


if __name__ == "__main__":
    run_snake()
