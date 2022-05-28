import numpy as np
import pygame



#this file is copied from https://github.com/davide97l/Snake-Battle-Royale/tree/master/snake
#It implements the player which implements the snake itself and how it moves



class Player(object):

    def __init__(self, game, color="green", ai="heuristic", depth=10):
        self.color = color
        
        self.image = pygame.image.load('./img/snakeBody1.png')
        x = 0.3 * game.game_width
        y = 0.3 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []  # coordinates of all the parts of the snake
        self.position.append([self.x, self.y])  # append the head
        self.food = 1  # length
        self.eaten = False
        self.right = 0
        self.left = 1
        self.up = 2
        self.down = 3
        self.direction = self.right
        self.step_size = 20  # pixels per step
        self.crash = False
        self.score = 0
        self.record = 0
        self.deaths = 0
        self.total_score = 0  # accumulated score
        self.agent = None
        self.ai = ai  # options: heuristic, deep_search
        if ai == "deepSearch":
            self.depth = depth  # search range for deep_search move

    def init_player(self, game):
        self.x, self.y = game.find_free_space()
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.direction = self.right
        self.step_size = 20
        self.crash = False
        self.score = 0

    def update_position(self):
        if self.position[-1][0] != self.x or self.position[-1][1] != self.y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = self.x
            self.position[-1][1] = self.y

    def eat(self, game):
        for food in game.food:
            if self.x == food.x_food and self.y == food.y_food:
                food.food_coord(game)
                self.eaten = True
                self.score += 1
                self.total_score += 1

    def crushed(self, game, x=-1, y=-1):
        if x == -1 and y == -1:  # coordinates of the head
            x = self.x
            y = self.y
        if x < 20 or x > game.game_width - 40 \
                or y < 20 or y > game.game_height - 40:
            return True
        
        if [x, y] in game.player.position:
            return True

    def do_move(self, move, game):
        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1

        move_array = [0, 0]

        if move == self.right:
            move_array = [self.step_size, 0]
        elif move == self.left:
            move_array = [-self.step_size, 0]
        elif move == self.up:
            move_array = [0, -self.step_size]
        elif move == self.down:
            move_array = [0, self.step_size]

        if move == self.right and self.direction != self.left:
            move_array = [self.step_size, 0]
            self.direction = self.right
        elif move == self.left and self.direction != self.right:
            move_array = [-self.step_size, 0]
            self.direction = self.left
        elif move == self.up and self.direction != self.down:
            move_array = [0, -self.step_size]
            self.direction = self.up
        elif move == self.down and self.direction != self.up:
            move_array = [0, self.step_size]
            self.direction = self.down
        self.x += move_array[0]
        self.y += move_array[1]

        if self.crushed(game):
            self.crash = True
            self.deaths += 1
            if self.score > self.record:
                self.record = self.score

        self.eat(game)
        self.update_position()

    def display_player(self, game):
        self.position[-1][0] = self.x
        self.position[-1][1] = self.y

        if not self.crash:
            for i in range(self.food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                #pygame.draw.rect(game.gameDisplay, (255,0,0), pygame.Rect(x_temp, y_temp, self.step_size, self.step_size))
                game.gameDisplay.blit(self.image, (x_temp, y_temp))
            pygame.display.update()
        else:
            game.end = True

    def move(self, game):
        distance = []
        for food in game.food:
            distance.append(abs(self.x - food.x_food) + abs(self.y - food.y_food))
        food = game.food[np.argmin(distance)]
        state = self.agent.get_state(game, self, food)
        prediction = self.agent.model(state)
        move = np.argmax(prediction[0])
        return move



    def move_as_array(self, move):
        if move == self.right:
            return [1, 0, 0, 0]
        elif move == self.left:
            return [0, 1, 0, 0]
        elif move == self.up:
            return [0, 0, 1, 0]
        elif move == self.down:
            return [0, 0, 0, 1]

    def set_agent(self, agent):
        self.agent = agent

    def distance_closest_food(self, game, x, y):
        distance = []
        for food in game.food:
            distance.append(abs(x - food.x_food/self.step_size) + abs(y - food.y_food/self.step_size))
        return distance[np.argmin(distance)]
