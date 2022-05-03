import random
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.signal import convolve   
import sys
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


    def move(self,keystroke):
        # predictions=self.decision_model(np.array([state]))[0]
        # index=np.where(predictions == np.amax(predictions))[0][0]
        # keystroke=[None,"left","right"][index]
        if keystroke=="left":
            self.move_left()
        elif keystroke=="right":
            self.move_right()
        elif keystroke=="up":
            self.move_up()
        elif keystroke=="down":
            self.move_down()
        return keystroke

    def calc_new_state(self,state,action):
        new_state=state.copy()
        custom_filter=np.zeros((3,3))
        filter_rule={"left":(0,1),"right":(2,1),"up":(1,0),"down":(1,2)}
        sel_filter_rule=filter_rule[action]
        custom_filter[sel_filter_rule[0]][sel_filter_rule[1]]=1
        new_state[1]=convolve(new_state[1],custom_filter,"same")
        #TODO: does not handle rest of body yet, only head
        return new_state

    def possible_states(self,state,counter=0):
        states=[]
        if counter==5:
            return states
        for action in ["up","down","left","right"]:
            new_state=self.calc_new_state(state,action)
            states.append(new_state)
            temp_states=self.possible_states(new_state,counter=counter+1)
            states.extend(temp_states)
        return states
    

    def move_complicated(self,state):
        states=self.possible_states(state)
        predictions=self.decision_model.predict(np.array(states))
        decisions=[]
        temp_state=state
        for _ in range(0,5):
            index_state=np.where(states==temp_state)
            predictions_state=predictions[0]
            index_max=np.where(predictions_state == np.amax(predictions_state))[0][0]
            decisions.append(["up","down","left","right"][index_max])
            temp_state=self.calc_new_state(temp_state,decisions[-1])
        return decisions
            
        




class Game:
    def __init__(self,size_game,snake):
        self.snake=snake
        self.size_game=size_game
        #self.distance_apple=self.calc_distance()
        self.state=self.init_state()
        self.apple_pos=None
        self.score=0
        self.update_apple()



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
        print(self.snake.pos[0])
        print(self.size_game)
        print(self.size_game<self.snake.pos[0][0] or self.snake.pos[0][0]<0 or self.size_game<self.snake.pos[0][1] or self.snake.pos[0][1]<0)
        if self.snake.pos[0] in self.snake.pos[1:]:
            return True
        #checks if snake has hit wall
        elif self.size_game<self.snake.pos[0][0] or self.snake.pos[0][0]<0 or self.size_game<self.snake.pos[0][1] or self.snake.pos[0][1]<0:
            return True
        return False
    
    def check_apple(self):
        """Checks if apple is where the snake is
        """
        state=self.state
        ate_apple=np.logical_and(state[0],state[1])
        if np.sum(ate_apple)==1:
            self.update_apple()
            self.snake.score+=20


    def update_apple(self):
        self.apple_pos=(random.randint(0,self.size_game-1),random.randint(0,self.size_game-1))
        self.state[0][self.apple_pos[0]][self.apple_pos[1]]=1

    def init_state(self):
        """Returns the state of the game

        Returns:
            list of lists: lists of lists with 0 for all points on map and 1 where snake is present and 2 where apple is
        """
        #one matrix each for apple,head,body
        state=np.zeros((3,self.size_game,self.size_game))
        snake_pos=self.snake.pos
        for index,snake_point in enumerate(snake_pos):
            state[2][snake_point[0]][snake_point[1]]=index+1
        state[1][snake_pos[0][0]][snake_pos[0][1]]=1
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
            decisions=self.snake.move_complicated(state=self.state)
            for action in decisions:
                self.snake.move(action)
                if self.check_loss():
                    print("broek")
                    self.snake.set_score(self.snake.get_score()-100)
                    break
                # self.distance_score()
                self.check_apple()
            #game_info.append((self.snake.pos.copy(),self.apple.pos,decision))
            #self.snake.game_info=game_info
        return game_info



#run this to save to file that can then be visualized
# gamey=Game(100,random_decision,snakey)
# print(gamey.get_state())
#info,score=gamey.run()
# results=pd.DataFrame(info)
# results.columns=["Pos. snake","Pos. apple","Decision"]
# results.to_csv("Random snake.csv")
