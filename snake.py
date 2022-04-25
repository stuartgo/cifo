import random
import time
import pandas as pd
import numpy as np
class Snake:
    def __init__(self,decision_model,size_game):
        self.pos=[(random.randint(0,size_game),random.randint(0,size_game))]
        self.len=len(self.pos)
        self.direction="up"
        self.decision_model=decision_model
    

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
    
        predictions=self.decision_model.predict(np.array([np.array(state)]))[0]
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


    def eat_apple(self):
        self.score+=1



class Apple:
    def __init__(self,size_game):
        self.size_game=size_game
        self.pos=(random.randint(0,size_game),random.randint(0,size_game))
    
    def move(self):
        self.pos=(random.randint(0,self.size_game),random.randint(0,self.size_game))


class Game:
    def __init__(self,size_game,snake):
        self.snake=snake
        self.apple=Apple(size_game)
        self.size_game=size_game
        self.score=0

    def check_loss(self):
        """Checks if snake has died

        Returns:
            _type_: _description_
        """
        if self.snake.pos[0] in self.snake.pos[1:]:
            return True
        elif self.size_game<self.snake.pos[0][0] or self.snake.pos[0][0]<0 or self.size_game<self.snake.pos[0][1] or self.snake.pos[0][1]<0:
            print("Wall")
            return True
        return False
    
    def check_apple(self):
        """Checks if apple is where the snake is
        """
        if self.snake.pos[0]==self.apple.pos:
            self.score+=1
    
    def get_state(self):
        """Returns the state of the game

        Returns:
            list of lists: lists of lists with 0 for all points on map and 1 where snake is present and 2 where apple is
        """
        state=[0]*self.size_game
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
                self.score-=100
                break
            self.check_apple()
            game_info.append((self.snake.pos.copy(),self.apple.pos,decision))
        return game_info,self.score

# def random_decision():
#     return ["left","right",None][random.randint(0,2)]
# snakey=Snake("ass",100)
# gamey=Game(100,random_decision,snakey)
# print(gamey.get_state())
# results=pd.DataFrame(gamey.run())
# results.columns=["Pos. snake","Pos. apple","Decision"]
# results.to_csv("Random snake.csv")
