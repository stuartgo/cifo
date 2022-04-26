from snake import Snake,Game
import random
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten
from tqdm import tqdm
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

    def run_generation(self):
        for snake in tqdm(self.snakes):
            game=Game(self.size_game,snake)
            info=game.run()
        
    def get_worst_snake(self):
        worst=self.snakes[0]
        for snake in self.snakes:
            if snake.score<worst.score:
                worst=snake
        return worst
    
    def get_sum_scores(self):
        sum_scores=0
        for snake in self.snakes:
            sum_scores+=snake.score
        return abs(sum_scores)
        

    def fitness_proportional(self):
        #TODO: The probabilities are very very low here so it basically never selects anything, fix?
        selected_snakes=[]
        #used to only get positive scores in the case of negative scores
        lowest_score=abs(self.get_worst_snake().score)
        for snake in self.snakes:
            #+1 is added to not get 0 scores
            probability=(snake.score+lowest_score+1)/abs(self.get_sum_scores())
            print(probability)
            if random.random()<=probability:
                selected_snakes.append(snake)
        return selected_snakes

    def ranking(self):
        selected_snakes=[]
        sorted_snakes=sorted(self.snakes,key=lambda snake: snake.score,reverse=False)
        for index,snake in enumerate(sorted_snakes):
            probability=(index+1)/len(sorted_snakes)
            if random.random()<=probability:
                selected_snakes.append(snake)
        return selected_snakes
    # def tournament():

snakes=Snake_Pop(100,50)

snakes.run_generation()
print(snakes.ranking())
