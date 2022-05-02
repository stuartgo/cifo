import random
import time
import pandas as pd
import numpy as np
import tensorflow as tf
class Snake:
    def __init__(self,decision_model,size_game,weights=None):
        self.pos=[(random.randint(0,size_game-1),random.randint(0,size_game-1))]
        self.len=len(self.pos)
        self.direction="up"
        self.decision_model=decision_model
        if not weights is None:
            for index,layer_weights in enumerate(weights):
                self.decision_model.layers[index].set_weights(layer_weights)
        self.score=0
        self.weights=self.get_weights()
        self.size_game=size_game
        self.game_info=None


    def get_weights(self):
        weights=[]
        for layer in self.decision_model.layers:
            weights.append(layer.get_weights())
        return weights

    def set_score(self,score):
        self.score=score
    
    def get_score(self):
        return self.score

    def move_up(self):
            """Move up refers to up on screen and not relative to snake
            """
            current_pos=self.pos[0]
            self.pos.pop()
            self.pos.insert(0,(current_pos[0],current_pos[1]+1))

    def move_down(self):
        """Move down refers to down on screen and not relative to snake
        """
        current_pos=self.pos[0]
        self.pos.pop()
        self.pos.insert(0,(current_pos[0],current_pos[1]-1))

    def move_right(self):
        """Move right refers to right on screen and not relative to snake
        """
        current_pos=self.pos[0]
        self.pos.pop()
        self.pos.insert(0,(current_pos[0]+1,current_pos[1]))

    def move_left(self):
        """Move left refers to left on screen and not relative to snake
        """
        current_pos=self.pos[0]
        self.pos.pop()
        self.pos.insert(0,(current_pos[0]-1,current_pos[1]))




    def move(self,state):
        predictions=self.decision_model(np.array([np.array(state)]))[0]
        index=np.where(predictions == np.amax(predictions))[0][0]
        keystroke=[None,"left","right"][index]
        if (keystroke=="left" and self.direction=="up") or (keystroke=="right" and self.direction=="down") or (self.direction=="left",keystroke==None):
            self.direction="left"
            self.move_left()
        elif (keystroke=="left" and self.direction=="down") or (keystroke=="right" and self.direction=="up") or (self.direction=="right",keystroke==None):
            self.direction="right"
            self.move_right()
        elif (keystroke=="left" and self.direction=="right") or (keystroke=="right" and self.direction=="left") or (self.direction=="up",keystroke==None):
            self.direction="up"
            self.move_up()
        elif (keystroke=="left" and self.direction=="left") or (keystroke=="right" and self.direction=="right") or (self.direction=="down",keystroke==None):
            self.direction="down"
            self.move_down()
        return keystroke




class Apple:
    def __init__(self,size_game):
        self.size_game=size_game
        self.pos=(random.randint(0,size_game-1),random.randint(0,size_game-1))
    
    def move(self):
        self.pos=(random.randint(0,self.size_game-1),random.randint(0,self.size_game-1))


class Game:
    def __init__(self,size_game,snake):
        self.snake=snake
        self.apple=Apple(size_game)
        self.size_game=size_game
        self.distance_apple=self.calc_distance()


    def calc_distance(self):
        """returns manhattan distance

        Returns:
            int: distance
        """
        distance= abs(self.snake.pos[0][0]-self.apple.pos[0])+abs(self.snake.pos[0][1]-self.apple.pos[1])
        return distance
    
    def distance_score(self):
        """Updates score based on where the snake moves, called after each movement
        """
        new_distance=self.calc_distance()
        if new_distance<self.distance_apple:
            self.snake.set_score(self.snake.get_score()+1)
        else:
            self.snake.set_score(self.snake.get_score()-1)
        self.distance_apple=new_distance





    def check_loss(self):
        """Checks if snake has died

        Returns:
            _type_: _description_
        """
        #checks if snake has hit itself
        if self.snake.pos[0] in self.snake.pos[1:]:
            return True
        #checks if snake has hit wall
        elif self.size_game<self.snake.pos[0][0] or self.snake.pos[0][0]<0 or self.size_game<self.snake.pos[0][1] or self.snake.pos[0][1]<0:
            return True
        return False
    
    def check_apple(self):
        """Checks if apple is where the snake is
        """
        if self.snake.pos[0]==self.apple.pos:
            self.snake.set_score(self.snake.get_score()+20)
            self.apple.move()
    
    def get_state(self):
        """Returns the state of the game

        Returns:
            list of lists: lists of lists with 0 for all points on map and 1 where snake is present and 2 where apple is
        """
        state=[0]*(self.size_game)
        state=[state.copy() for _ in range(self.size_game)]
        for snake_point in self.snake.pos:
            state[snake_point[0]][snake_point[1]]=1
        apple=self.apple.pos
        state[apple[0]][apple[1]]=2
        return state

    def run(self):
        """Runs a game until the snake fails

        Returns:
            list of tuples: the position of all the "blocks" that make up the snake
            tuple: position of the current apple
            string: the choice taken at this step in the game
        """
        game_info=[]
        while True:
            decision=self.snake.move(state=self.get_state())
            if self.check_loss():
                self.snake.set_score(self.snake.get_score()-100)
                break
            self.distance_score()
            self.check_apple()
            game_info.append((self.snake.pos.copy(),self.apple.pos,decision))
            self.snake.game_info=game_info
        return game_info



#run this to save to file that can then be visualized
gamey=Game(100,random_decision,snakey)
print(gamey.get_state())
info,score=gamey.run()
results=pd.DataFrame(info)
results.columns=["Pos. snake","Pos. apple","Decision"]
results.to_csv("Random snake.csv")
