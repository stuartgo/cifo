from cifo1 import Snake, Game
import numpy as np
import pandas as pd
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tqdm import tqdm
import sys

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
        model.add(Flatten(input_shape = (size_game, size_game)))
        model.add(Dense(8, kernel_initializer = "random_uniform", activation = "relu"))
        model.add(Dense(3, kernel_initializer = "random_uniform", activation = "softmax"))

        # create a population of snake objects
        self.snakes = [Snake(size_game, keras.models.clone_model(model)) for _ in range(pop_size)]

        # initialize population attributes
        self.size_game = size_game
        self.pop_size = pop_size

    
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
            print("Evaluating snakes...")
            for snake in tqdm(self.snakes):
                game = Game(self.size_game, snake)
                game.run()

            # creating new population
            new_pop = []
            
            while len(new_pop) < self.pop_size:
                # apply crossover with a given probability (p_crossover)
                if random.random() <= p_crossover:
                    genetic_operator = "crossover"
                else:
                    genetic_operator = "reproduction"

                # select snakes according to the chosen selection method
                snake_pair = self.tournament(50)[:2]

                # genetic_operator = "reproduction" # TODO: Delete this line when the crossover issue has been fixed

                if genetic_operator == "crossover":
                    offspring = self.n_point_standard_crossover(snake_pair[0], snake_pair[1])
                else:
                    offspring = snake_pair
                
                mutated_offspring = []
                for child in offspring:
                    mutated_offspring.append(self.standard_mutation(child, p_mutation))
                new_pop.extend(mutated_offspring)

            scores = self.get_scores()
            print(scores, sum(scores)/len(scores))
            best = self.get_best_snake()
            results = pd.DataFrame(best.game_info)
            results.columns = ["Pos. snake", "Pos. apple", "Decision"]
            results.to_csv("./Snakes/Best Snake gen" + str(counter) + ".csv")

            self.snakes = new_pop
            counter += 1


    def get_worst_snake(self):

        worst_snake = self.snakes[0]

        for snake in self.snakes:
            if snake.score < worst_snake.score:
                worst_snake = snake

        return worst_snake

    
    def get_best_snake(self):

        best_snake = self.snakes[0]

        for snake in self.snakes:
            if snake.score > best_snake.score:
                best_snake = snake

        return best_snake


    def get_scores(self):

        scores = []

        for snake in self.snakes:
            scores.append(snake.score)

        return sorted(scores)

    
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
            abs_worst_score = self.get_worst_snake().score + 1

            # sum fitness of all snakes in the population
            total_score = sum([snake.score + abs_worst_score for snake in self.snakes])

            # get a position on the wheel
            spin = random.uniform(0, total_score)
            position = 0

            # find the individual in the landing position
            for snake in self.snakes:
                updated_score = snake.score + abs_worst_score
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
        sorted_snakes = sorted(self.snakes, key = lambda snake: snake.score, reverse = False)
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
        sorted_snakes=sorted(random_sample, key = lambda snake: snake.score, reverse = True)
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
                    weights_flat[index]=random.uniform(min_val,max_val)
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
        return [Snake(snake1.size_game, snake1.decision_model,weights=new_weights[0]),Snake(snake1.size_game, snake1.decision_model,weights=new_weights[1])]
    

    def n_point_standard_crossover(self,snake1,snake2):
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
            new_weights_temp1=np.zeros(len(weights_flat1))
            new_weights_temp2=np.zeros(len(weights_flat1))
            for index,k in enumerate(weights_flat1):
                if random.random()<=0.5:
                    new_weights_temp1[index]=k
                    new_weights_temp2[index]=weights_flat2[index]
                else:
                    new_weights_temp2[index]=k
                    new_weights_temp1[index]=weights_flat2[index]
            new_weights[0].append([np.reshape(new_weights_temp1,original_shape),layer_weight1[1]])
            new_weights[1].append([np.reshape(new_weights_temp2,original_shape),layer_weight1[1]])
        return [Snake(snake1.size_game, snake1.decision_model,weights=new_weights[0]),Snake(snake1.size_game, snake1.decision_model,weights=new_weights[1])]


snakes = SnakePop(500, 10, optim = "max")

snakes.evolve(1000, 0.75, 0.05)