from sqlite3 import complete_statement
from snake import Snake,Game
import random
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten
from tqdm import tqdm
import numpy as np
import pandas as pd
# Neural network
# model = Sequential()
# model.add(Dense(16, input_dim=100,kernel_initializer="random_uniform", activation="relu"))
# model.add(Dense(3, activation="softmax",kernel_initializer="random_uniform",))
import multiprocessing as mp


class Snake_Pop:

    def __init__(self,pop_size,size_game):
        model = Sequential()
        model.add(Flatten(input_shape=(size_game,size_game))),
        model.add(Dense(8,kernel_initializer="random_uniform", activation="relu"))
        model.add(Dense(3, activation="softmax",kernel_initializer="random_uniform"))
        self.snakes=[Snake(keras.models.clone_model(model),size_game) for _ in range(pop_size)]
        self.size_game=size_game
        self.pop_size=pop_size

    def run_generation(self,prob_crossover,mutation_rate):
        counter=0
        while counter<1000:
            #calculate the fitness for each snake
            print("Evaluating snakes")
            for snake in tqdm(self.snakes):
                game=Game(self.size_game,snake)
                game.run()
            #creating new population
            new_pop=[]
            while len(new_pop)!=self.pop_size:
                #selecting genetic operator
                random_val=random.random()
                if random_val<=prob_crossover:
                    genetic_operator="crossover"
                else:
                    genetic_operator="reproduction"
                #select two snakes
                two_snakes=self.ranking()[:2]
                if genetic_operator=="crossover":
                    offspring=self.standard_crossover(two_snakes[0],two_snakes[1])
                else:
                    offspring=two_snakes
                mutated_offspring=[]
                for child in offspring:
                    mutated_offspring.append(self.standard_mutation(child,mutation_rate))
                new_pop.extend(mutated_offspring)
            scores=self.get_scores()
            print(scores, sum(scores)/len(scores))
            best=self.get_best_snake()
            results=pd.DataFrame(best.game_info)
            results.columns=["Pos. snake","Pos. apple","Decision"]
            results.to_csv("Best Snake gen"+str(counter)+".csv")
                
            self.snakes=new_pop
            counter+=1


    def get_worst_snake(self):
        worst=self.snakes[0]
        for snake in self.snakes:
            if snake.score<worst.score:
                worst=snake
        return worst
    
    def get_best_snake(self):
        best=self.snakes[0]
        for snake in self.snakes:
            if snake.score>best.score:
                best=snake
        return best
    def get_scores(self):
        scores=[]
        for snake in self.snakes:
            scores.append(snake.score)
        return sorted(scores)
    

    def get_sum_scores(self):
        sum_scores=0
        for snake in self.snakes:
            sum_scores+=snake.score
        return abs(sum_scores)
        
    ###
    #Selection methods
    ###
    def fitness_proportional(self):
        """Returns snakes based on fitness score, the higher the fitness the higher the probability of being selected.

        Returns:
            list of snakes: -
        """
        #TODO: The probabilities are very very low here so it basically never selects anything, fix?
        selected_snakes=[]
        #used to only get positive scores in the case of negative scores
        lowest_score=abs(self.get_worst_snake().score)
        for snake in self.snakes:
            #+1 is added to not get 0 scores
            probability=(snake.score+lowest_score+1)/abs(self.get_sum_scores())
            if random.random()<=probability:
                selected_snakes.append(snake)
        return selected_snakes

    def ranking(self):
        """Returns snakes based on ranking, the higher the fitness the higher the probability of being selected. Selection is based on position relative to other snakes

        Returns:
            list of snakes: -
        """
        selected_snakes=[]
        sorted_snakes=sorted(self.snakes,key=lambda snake: snake.score,reverse=False)
        for index,snake in enumerate(sorted_snakes):
            probability=(index+1)/len(sorted_snakes)
            if random.random()<=probability:
                selected_snakes.append(snake)
        return selected_snakes
    
    
    def tournament(self,tournament_size):
        """This method selects a single individual

        Args:
            tournament_size (int): Number of snakes randonly selected
        """
        random_sample=random.sample(self.snakes,tournament_size)
        best_snake=random_sample[0]
        for snake in random_sample:
            if snake.score>=best_snake:
                best_snake=snake
        return best_snake
    
    ###
    #Mutation operators
    ###
    def standard_mutation(self,snake,mutation_rate):
        weights=snake.get_weights()
        new_weights=[]
        #iterates over the weights for each layer in model
        for layer_weight in weights:
            #first layer is empty as its a flattening layer
            if layer_weight==[]:
                new_weights.append([])
                continue
            #TODO: not entirely sure if I understood what the second array is
            #the weights have a secondary array that is not to be modified?
            second_array=layer_weight[1]
            #store shape and flatten
            original_shape=np.array(layer_weight[0]).shape
            weights_flat=np.array(layer_weight[0]).flatten()
            #using min and max to define what is an admissable value
            min_val=np.amin(weights_flat)
            max_val=np.amax(weights_flat)
            #modify each weight with probability mutation rate
            #if modified a random value between min and max value of the existing weights is used
            for index,gene in enumerate(weights_flat):
                if random.random()<=mutation_rate:
                    weights_flat[index]=random.uniform(min_val,max_val)
            new_weights.append([np.reshape(weights_flat,original_shape),second_array])
        return Snake(snake.decision_model,snake.size_game,weights=new_weights)
    

    ###
    #Crossover operators
    ###
    def standard_crossover(self,snake1,snake2):
        #as we are using multiple layers the crossover is repeated for each layer
        weights1=snake1.weights
        weights2=snake2.weights
        #one list for the weights of each of the two offspring
        new_weights=[[],[]]
        for layer_num in range(len(weights1)):
            layer_weight1=weights1[layer_num]
            layer_weight2=weights2[layer_num]
            if layer_weight1==[]:
                new_weights[0].append([])
                new_weights[1].append([])
                continue
            original_shape=np.array(layer_weight1[0]).shape
            weights_flat1=np.array(layer_weight1[0]).flatten()
            weights_flat2=np.array(layer_weight2[0]).flatten()
            crossover_point=random.randint(0,len(weights_flat1))
            complete_flat1=np.hstack([weights_flat1[:crossover_point],weights_flat2[crossover_point:]])
            complete_flat2=np.hstack([weights_flat2[:crossover_point],weights_flat1[crossover_point:]])
            new_weights[0].append([np.reshape(complete_flat1,original_shape),layer_weight1[1]])
            new_weights[1].append([np.reshape(complete_flat2,original_shape),layer_weight1[1]])
        return [Snake(snake1.decision_model,snake1.size_game,weights=new_weights[0]),Snake(snake1.decision_model,snake1.size_game,weights=new_weights[1])]

    

    
snakes=Snake_Pop(100,20)

snakes.run_generation(0.5,0.15)

