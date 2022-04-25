from snake import Snake,Game
import keras

import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten
# Neural network
# model = Sequential()
# model.add(Dense(16, input_dim=100,kernel_initializer="random_uniform", activation="relu"))
# model.add(Dense(3, activation="softmax",kernel_initializer="random_uniform",))



class Snake_Pop:

    def __init__(self,pop_size,size_game):
        model = Sequential()
        model.add(Flatten(input_shape=(size_game,size_game))),
        model.add(Dense(16,kernel_initializer="random_uniform", activation="relu"))
        model.add(Dense(3, activation="softmax",kernel_initializer="random_uniform"))
        self.snakes=[Snake(model,size_game) for _ in range(pop_size)]
        self.size_game=size_game
        self.scores=[]

    def run_generation(self):
        for snake in self.snakes:
            game=Game(self.size_game,snake)
            info,score=game.run()
            self.scores.append(score)
            
        

    # def fitness_proportional():


    # def ranking():

    # def tournament():

snakes=Snake_Pop(3,50)

snakes.run_generation()
print(snakes.scores)