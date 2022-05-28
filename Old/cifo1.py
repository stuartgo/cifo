import random
from matplotlib.ft2font import HORIZONTAL
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import time
class Snake:
    """
    A class used to represent a snake.

    ...

    Attributes
    ----------
    size_game: int
        size of the game board (e.g., 15)
    init: tuple
        initial position of the snake (e.g., (4, 6))
        assigned randomly if left unspecified
    representation

    optim: str
        
    """

    def __init__(
        self,
        size_game,
        decision_model,
        init_x = None,
        init_y = None,
        weights = None,
    ):

        # assign random snake spawn location if initial location is not specified
        if (init_x is None) and (init_y is None):
            self.pos = [(
                random.randint(0, size_game-1),
                random.randint(0, size_game-1)
            )]
        # spawn snake at the specified location if it is inside the screen
        elif (init_x >= 0) and (init_x < size_game-1) and (init_y >= 0) and (init_y < size_game-1):
            self.pos = [(init_x, init_y)]
        # raise exception if specified spawning location is off screen
        else:
            raise Exception("init_x and init_y must be integers in the interval [0, size_game-1]")

        # initialize snake attributes
        self.game_info=None
        self.len = len(self.pos)
        self.direction = "up"
        self.score = 0
        self.decision_model = decision_model
        self.size_game = size_game
        self.fitness=None
        self.record=0
        self.total_apples_eaten=0
        self.grow=False

        # if weights passed as argument, assign each weight to a layer of the neural net
        if weights is not None:
            for index, layer_weights in enumerate(weights):
                self.decision_model.layers[index].set_weights(layer_weights)
        # get weights of the layers
        if self.decision_model=="random":
            self.weights=None
        else:
            self.weights = self.get_weights()


    def get_weights(self):
        """
        Retrieves weights of the layers in the neural network.
        """

        weights = []

        for layer in self.decision_model.layers:
            weights.append(layer.get_weights())
        return weights

    def new_pos(self):
        self.pos = [(
                random.randint(0, self.size_game-1),
                random.randint(0, self.size_game-1)
            )]
        self.grow=False
        




    def calc_fitness(self):
        #This code is quite bad

        run_info=self.game_info
        try:
            penalties=run_info.iloc[:,0].value_counts()["penalty"]
        except:
            penalties=0
        try:
            deaths=dict(run_info.iloc[:,0].value_counts())["death"]
        except:
            deaths=0

        num_steps=len(run_info)/(self.total_apples_eaten+1)
        # if 0<self.record*5000-deaths*150-num_steps*100-penalties*1000:
        #     print("LAAAAAAAAAAArger than 0")
        #     print(self.record,deaths,num_steps,penalties)
        self.fitness=self.record*5000-deaths*150-num_steps*100-penalties*1000
            

        

    def move(self, state):
        """
        Updates the position of the snake's head and body
        according to the direction it's moving in.

        Returns
        -------
        keystroke: str or None
            direction in which the snake is moving if turning
            None if the snake's direction is unchanged

        """

        # determine direction in which to move based on the output of the decision model
        if self.decision_model=="random":
            keystroke=random.choice([None, "left", "right"])
        else:
            predictions = self.decision_model(np.array([state]))[0]
            index = np.where(predictions == np.amax(predictions))[0][0]
            keystroke = [None, "left", "right"][index]

        # store current position of the head
        current_pos = self.pos[0]

        if not self.grow:
            self.pos.pop()
        else:
            self.grow=False
        # remove last position on the list (snake's tail)
        
        # update position of the snake based on moving direction
        if (
            (keystroke == "left" and self.direction == "up")
            or (keystroke == "right" and self.direction == "down")
            or (self.direction == "left" and keystroke == None)
        ):
            self.direction = "left"
            self.pos.insert(0, (current_pos[0]-1, current_pos[1]))
            
        elif (
            (keystroke == "left" and self.direction == "down")
            or (keystroke == "right" and self.direction == "up")
            or (self.direction == "right" and keystroke == None)
        ):
            self.direction = "right"
            self.pos.insert(0, (current_pos[0]+1, current_pos[1]))
        elif (
            (keystroke == "left" and self.direction == "right")
            or (keystroke == "right" and self.direction == "left")
            or (self.direction == "up" and keystroke == None)
        ):
            self.direction = "up"
            self.pos.insert(0, (current_pos[0], current_pos[1]+1))
        elif (
            (keystroke == "left" and self.direction == "left")
            or (keystroke == "right" and self.direction == "right")
            or (self.direction == "down" and keystroke == None)
        ):
            self.direction = "down"
            self.pos.insert(0, (current_pos[0], current_pos[1]-1))
        
        return keystroke
        




class Game:
    """
    A class to represent each game.
    
    ...

    Attributes
    ----------
    size_game: int
        size of the game board (e.g., 20) 
    snake: ????????
        ?????????????????????

    """

    def __init__(self, size_game, snake):
        self.snake = snake
        self.apple = (
                random.randint(0, size_game-1),
                random.randint(0, size_game-1)
            )
        self.size_game = size_game


    def move_apple(self):
        crash_snake=True
        while crash_snake is True:
            temp_apple = (
                    random.randint(0, self.size_game-1),
                    random.randint(0, self.size_game-1)
                )
            if temp_apple in self.snake.pos:
                crash_snake=True
            else:
                crash_snake=False
        self.apple=temp_apple

    def check_loss(self):
        """
        Checks for game losing conditions.

        """

        # check if snake has hit itself
        if self.snake.pos[0]==self.snake.pos[1:]:
            return True

        # check if snake has hit a wall (game border)
        if (
            (self.size_game <= self.snake.pos[0][0])
            or (self.size_game <= self.snake.pos[0][1])
            or (self.snake.pos[0][0] < 0)
            or (self.snake.pos[0][1] < 0)
        ):
            return True
        
        return False


    def check_apple(self):
        """
        Checks if the apple is at the same position
        as the snake's head.
        """
        
        if self.snake.pos[0] == self.apple:
            self.move_apple()
            self.snake.grow=True
            self.snake.total_apples_eaten+=1
            return True
        return False

    def get_state(self):
        """
        Returns the state of the ongoing game.

        Returns
        -------
        state: list of lists
            0 for all points on map, 1 where snake is present and 2 where apple is??????
        """

        state = np.zeros((self.size_game,self.size_game))

        for index,snake_point in enumerate(self.snake.pos):
            state[snake_point[0]][snake_point[1]] = index+1
        apple = self.apple
        state[apple[0]][apple[1]] = -1

        return state

    
    def get_state2(self):
        #this is to avoid a bug i cant figure out how to fix
        if self.apple==self.snake.pos[0]:
            self.move_apple()
        state_matrix=self.get_state()
        head=np.where(state_matrix==1)
        # print(state_matrix)
        apple=np.where(state_matrix==-1)
        head=(head[0][0],head[1][0])
        state_matrix[apple[0][0]][apple[1][0]]=0
        apple=(apple[0][0],apple[1][0])
        #horizontal
        horizontal=state_matrix[head[0]]
        vertical=state_matrix[:][head[1]]
        for direction in ["left","right","up","down"]:
            if direction=="left":
                info_array=horizontal[:head[0]]
                dist=head[0]
            elif direction=="right":
                info_array=horizontal[head[0]+1:]
                dist=len(horizontal)-head[0]-1
            elif direction=="up":
                info_array=vertical[:head[0]]
                dist=head[0]
            elif direction=="down":
                info_array=vertical[head[0]+1:]
                dist=len(vertical)-head[0]-1
            for index,val in enumerate(info_array):
                if val!=0:
                    dist=index
            if direction=="left":
                dist_left=dist
            elif direction=="right":
                dist_right=dist
            elif direction=="up":
                dist_up=dist
            elif direction=="down":
                dist_down=dist
        if apple[0]>head[0]:
            applex=1
        elif apple[0]==head[0]:
            applex=0
        elif apple[0]<head[0]:
            applex=-1
        if apple[1]>head[1]:
            appley=-1
        elif apple[1]==head[1]:
            appley=0
        elif apple[1]<head[1]:
            appley=1
        #returns distance blocked in front, to the left, to the right(all relative to the snake)
        #direction of movement and direction for apple
        #left,right,infront,applex,appley,snakex,snakey
        direction=self.snake.direction
        if direction=="up":
            return np.array([dist_left,dist_right,dist_up,applex,appley,0,1])
        elif direction=="down":
            return np.array([dist_right,dist_left,dist_down,applex,appley,0,-1])
        elif direction=="left":
            return np.array([dist_down,dist_up,dist_left,applex,appley,-1,0])
        elif direction=="right":
            return np.array([dist_up,dist_down,dist_right,applex,appley,1,0])

    def reset_game(self):
        self.snake.new_pos()
        self.move_apple()

    def run(self):
        """
        Run a game until the snake fails.

        Returns
        -------
        game_info: list of tuples, tuple, str
            list of tuples: position of all the blocks that make up the snake
            tuple: position of the game's last apple
            str: final choice made by the snake
        """
        game_info = []
        self.snake.new_pos()
        runs=0
        no_food=0
        run_record=0
        for _ in tqdm(range(0,5000),leave=False):
            if self.check_loss():
                game_info.append(("death","death", "death", "death","death"))
                self.reset_game()
                no_food=0
                runs+=1
                run_record=0
                continue
            if no_food==100:
                game_info.append(("penalty","penalty", "penalty", "penalty","penalty"))
                self.reset_game()
                no_food=0
                run_record=0
                continue
            state=self.get_state2()
            decision = self.snake.move(state = state )
            if self.check_apple():
                no_food=0
                run_record+=1
                if run_record> self.snake.record:
                    self.snake.record=run_record
            else:
                no_food+=1
            game_info.append((runs,self.snake.pos.copy(), self.apple, decision,*state,run_record))

        game_info=pd.DataFrame(game_info)
        game_info.columns=["run","Pos. snake","Pos. apple","Decision","left","right","front","applex","appley","snakex","snakey","score"]
        self.snake.game_info=game_info
    
        return game_info



#run this to save to file that can then be visualized


# snakey=Snake(10,"random")
# game = Game(10, snakey)
# game_info=game.run()
# snakey.calc_fitness()
# print(snakey.fitness)
# snakey.calc_fitness()
# print(snakey.fitness)

# game_info.to_csv("Random snake.csv")
# print("done")