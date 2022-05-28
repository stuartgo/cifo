from cifo1 import Snake, Game
import numpy as np
import pandas as pd
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tqdm import tqdm
import sys
from tensorflow.keras.initializers import RandomUniform

class SnakePop:
    """
    Create a population of snake objects.

    ...

    Attributes
    ----------
    pop_size: int
        size of the snake population (e.g., 500)
    size_game: int
        size of the game board (e.g., 20)
    optim: str
        type of optimization problem ("max" or "min")
    """

    def __init__(self, pop_size, size_game, optim):
        
        # create the neural network with 1 hidden layer
        # output layer with 3 neurons for 3 possible decisions: left, right, none
        model = Sequential()
        #model.add(Flatten(input_shape = (size_game, size_game)))
        initializer = RandomUniform(minval=-1, maxval=1)
        model.add(Dense(7, input_dim=7, activation="relu",kernel_initializer=initializer))
        model.add(Dense(120, activation="relu",kernel_initializer=initializer))
        model.add(Dense(3, kernel_initializer = initializer, activation = "softmax"))

        # create a population of snake objects
        self.snakes = [Snake(size_game, keras.models.clone_model(model)) for _ in range(pop_size)]

        # initialize population attributes
        self.size_game = size_game
        self.pop_size = pop_size
        self.optim=optim

    
    def evolve(self, gens, p_crossover, p_mutation, select = None, elitism = None):
        """
        Create new populations of snakes based on 
        the specified arguments.

        ...

        Attributes
        ----------
        gens: int
            number of generations
        select: str
            method used to select snakes of the original population
            possible values: "fps", "ranking", "tournament"
        p_crossover: float
            probability of crossover
        p_mutation: float
            probability of mutation
        elitism: bool
            if True, best scoring snake will be automatically transferred to new population
        """

        counter = 0

        while counter < gens:
            # run a game for each snake and determine fitness
            print("#####################")
            print("Generation:",counter)
            print("Evaluating snakes...")
            for snake in tqdm(self.snakes,leave=False):
                game = Game(self.size_game, snake)
                game.run()
                snake.calc_fitness()

            # creating new population
            new_pop = []
            print("Breeding high quality snakes")
            
            while len(new_pop) < self.pop_size:
                # apply crossover with a given probability (p_crossover)
                if random.random() <= p_crossover:
                    genetic_operator = "crossover"
                    selected_snakes=[self.fps(),self.fps()]
                    if selected_snakes[0]== selected_snakes[1]:
                        continue
                else:
                    genetic_operator = "reproduction"
                    selected_snakes=[self.fps()]

                # select snakes according to the chosen selection method

                # genetic_operator = "reproduction" # TODO: Delete this line when the crossover issue has been fixed

                if genetic_operator == "crossover":
                    offspring = self.geometric_semantic_crossover(selected_snakes[0], selected_snakes[1])
                    
                else:
                    offspring = selected_snakes[0]
                    #now adding the best ones directly in new generation in addition to their mutated offspring
                    

                mutated_offspring=self.geometric_semantic_mutation(offspring,0.025)
                new_pop.append(mutated_offspring)
                if elitism:
                    new_pop.append(self.get_best_snake())
                # game = Game(self.size_game, mutated_offspring)
                # game.run()
                # mutated_offspring.calc_fitness()
                # #This ensures that a mutated offspring cannot be worse than the average of its parent
                # # tqdm.write(str(mutated_offspring.fitness))
                # # tqdm.write(str(sum(list(map(lambda x: x.fitness,selected_snakes)))/len(selected_snakes)))
                # if mutated_offspring.fitness>=sum(list(map(lambda x: x.fitness,selected_snakes)))/len(selected_snakes):
                #     new_pop.append(mutated_offspring)
                #     pbar.update(len(new_pop))
                
            

            fitness_values = self.get_fitness()
            best = self.get_best_snake()
            print("fitness scores:",fitness_values)
            print("Average fitness:",sum(fitness_values)/len(fitness_values))
            print("Most apples eaten:",best.record)
            results = best.game_info
            results.to_csv("./Snakes/Best Snake gen" + str(counter) + ".csv")

            self.snakes = new_pop
            counter += 1


    def get_worst_snake(self):

        worst_snake = self.snakes[0]

        for snake in self.snakes:
            if snake.fitness < worst_snake.fitness:
                worst_snake = snake

        return worst_snake

    
    def get_best_snake(self):

        best_snake = self.snakes[0]

        for snake in self.snakes:
            if snake.fitness > best_snake.fitness:
                best_snake = snake

        return best_snake


    def get_fitness(self):

        fitness_values = []

        for snake in self.snakes:
            fitness_values.append(snake.fitness)

        return sorted(fitness_values)

    
    # ----------------- #
    # SELECTION METHODS #
    # ----------------- #

    def fps(self):
        """
        Fitness proportionate selection. Select snakes based on their fitness score.
        A higher fitness score means a higher probability of being selected.

        Returns:
            selected_snake (???): snake selected by the method 
        """

        if self.optim == "max":
            # absolute value of the worst score in the population to prevent negative values
            # added 1 to ensure only positive, non-null values
            abs_worst_fitness = abs(self.get_worst_snake().fitness) + 1

            # sum fitness of all snakes in the population
            total_score = sum([snake.fitness + abs_worst_fitness for snake in self.snakes])

            # get a position on the wheel
            spin = random.uniform(0, total_score)
            position = 0

            # find the individual in the landing position
            snakes=self.snakes.copy()
            random.shuffle(snakes)
            for snake in snakes:
                updated_score = snake.fitness + abs_worst_fitness
                position += updated_score

                if position > spin:
                    return snake

        elif self.optim == "min":
            raise NotImplementedError

        else:
            raise Exception("No optimization specified (min or max).")


    def ranking(self):
        """
        Select snakes based on their fitness score ranking.

        Returns:
            selected_snake (???): snake selected by the method
        """

        selected_snakes = []
        # sort snakes according to their score values
        sorted_snakes = sorted(self.snakes, key = lambda snake: snake.fitness, reverse = False)
        # add snakes to the selected_snakes list
        # snakes with a higher ranking have a higher probability of being in the list
        for index, snake in enumerate(sorted_snakes):
            probability = (index + 1) / len(sorted_snakes)
            if random.random() <= probability:
                selected_snakes.append(snake)
        
        # shuffles and returns selected snakes in random order
        random.shuffle(selected_snakes)
        return selected_snakes

    
    def tournament (self, size):
        """
        Select snakes based on tournament selection.

        Args:
            tournament_size (int): number of snakes randomly selected for tournament
        Returns:
            best_snake (???): snake who won the tournament
        """

        random_sample = random.sample(self.snakes, size)
        sorted_snakes=sorted(random_sample, key = lambda snake: snake.fitness, reverse = True)
        return sorted_snakes


    # ------------------ #
    # MUTATION OPERATORS #
    # ------------------ #

    def standard_mutation(self, snake, p_mutation):

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
                if random.random()<=p_mutation:
                    weights_flat[index]=random.uniform(-1,1)     #random.uniform(min_val,max_val)
            new_weights.append([np.reshape(weights_flat,original_shape),second_array])

        return Snake(snake.size_game, snake.decision_model,weights=new_weights)

    def geometric_semantic_mutation(self,snake,ms):
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

            for index,gene in enumerate(weights_flat):
                weights_flat[index]=gene+random.uniform(-ms,ms)
            new_weights.append([np.reshape(weights_flat,original_shape),second_array])

        return Snake(snake.size_game, snake.decision_model,weights=new_weights)

    
    # ------------------- #
    # CROSSOVER OPERATORS #
    # ------------------- #

    def standard_crossover(self,snake1,snake2):
        #as we are using multiple layers the crossover is repeated for each layer
        weights1=snake1.weights
        weights2=snake2.weights
        #one list for the weights of each of the two offspring
        new_weights=[]
        for layer_num in range(len(weights1)):
            layer_weight1=weights1[layer_num]
            layer_weight2=weights2[layer_num]
            if layer_weight1==[]:
                new_weights.append([])
                continue
            original_shape=np.array(layer_weight1[0]).shape
            weights_flat1=np.array(layer_weight1[0]).flatten()
            weights_flat2=np.array(layer_weight2[0]).flatten()
            crossover_point=random.randint(0,len(weights_flat1))
            complete_flat=np.hstack([weights_flat1[:crossover_point],weights_flat2[crossover_point:]])
            new_weights.append([np.reshape(complete_flat,original_shape),layer_weight1[1]])
        
        return Snake(snake1.size_game, snake1.decision_model,weights=new_weights)
    

    def uniform_crossover(self,snake1,snake2):
        #as we are using multiple layers the crossover is repeated for each layer
        weights1=snake1.weights
        weights2=snake2.weights
        #one list for the weights of each of the two offspring
        new_weights=[]
        for layer_num in range(len(weights1)):
            layer_weight1=weights1[layer_num]
            layer_weight2=weights2[layer_num]
            if layer_weight1==[]:
                new_weights.append([])
                continue
            original_shape=np.array(layer_weight1[0]).shape
            weights_flat1=np.array(layer_weight1[0]).flatten()
            weights_flat2=np.array(layer_weight2[0]).flatten()
            new_weights_temp=np.zeros(len(weights_flat1))
            for index,k in enumerate(weights_flat1):
                if random.random()<=0.5:
                    new_weights_temp[index]=k
                else:
                    new_weights_temp[index]=weights_flat2[index]
            new_weights.append([np.reshape(new_weights_temp,original_shape),layer_weight1[1]])
        return Snake(snake1.size_game, snake1.decision_model,weights=new_weights)
    

    def geometric_semantic_crossover(self,snake1,snake2):
        #as we are using multiple layers the crossover is repeated for each layer
        weights1=snake1.weights
        weights2=snake2.weights
        #one list for the weights of each of the two offspring
        new_weights=[]
        for layer_num in range(len(weights1)):
            layer_weight1=weights1[layer_num]
            layer_weight2=weights2[layer_num]
            if layer_weight1==[]:
                new_weights.append([])
                continue
            original_shape=np.array(layer_weight1[0]).shape
            weights_flat1=np.array(layer_weight1[0]).flatten()
            weights_flat2=np.array(layer_weight2[0]).flatten()
            new_weights_temp=np.zeros(len(weights_flat1))
            for index,k in enumerate(weights_flat1):
                r=random.random()
                x=k
                y=weights_flat2[index]
                new_weights_temp[index]=r*x+(1-r)*y
            new_weights.append([np.reshape(new_weights_temp,original_shape),layer_weight1[1]])
        return Snake(snake1.size_game, snake1.decision_model,weights=new_weights)


snakes = SnakePop(50, 10, optim = "max")

snakes.evolve(50, 0.75, 0.01,elitism=True)