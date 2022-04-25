import random
import time
import pandas as pd
class Snake:
    def __init__(self,init_pos,decision_algorithm):
        self.pos=[init_pos]
        self.len=len(self.pos)
        self.direction="up"
        self.decision_algorithm=decision_algorithm
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

    def move(self):
        keystroke=self.decision_algorithm()
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
    def __init__(self,size_game,decision_algorithm):
        self.snake=Snake((random.randint(0,size_game),random.randint(0,size_game)),decision_algorithm)
        self.apple=Apple(size_game)
        self.size_game=size_game
        self.score=0
        #Not sure if this is the best way to implement it, but it should allow you to implement various decision making processes for each snake
        self.choice_algorithm="random"

    def check_loss(self):
        if self.snake.pos[0] in self.snake.pos[1:]:
            return True
        elif self.size_game<self.snake.pos[0][0] or self.snake.pos[0][0]<0 or self.size_game<self.snake.pos[0][1] or self.snake.pos[0][1]<0:
            print("Wall")
            return True
        return False
    
    def check_apple(self):
        if self.snake.pos[0]==self.apple.pos:
            self.score+=1
    

    def run(self):
        game_info=[]
        while True:
            decision=self.snake.move()
            if self.check_loss():
                break
            self.check_apple()
            game_info.append((self.snake.pos.copy(),self.apple.pos,decision))
        return game_info

def random_decision():
    return ["left","right",None][random.randint(0,2)]

gamey=Game(100,random_decision)

results=pd.DataFrame(gamey.run())
results.columns=["Pos. snake","Pos. apple","Decision"]
results.to_csv("Random snake.csv")
