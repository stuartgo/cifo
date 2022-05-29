import random

from keras import backend as K
import os
from Game import Game
from Food import Food
from Player import Player
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import sys
import os

#this file contains some code taken from https://github.com/davide97l/Snake-Battle-Royale/tree/master/snake
#All the code pertaining to the genetic operators is written by us





#represents the decision making agent
class GAAgent(object):

    def __init__(self, units=[12, 32, 4], population_name="standard_population", generation=100,
                 population=50, training=False):
        
        self.units = units
        self.population_name = population_name
        self.generation = generation
        self.population = population
        if not training:
            self.model = create_model_from_units(units, best_snake_weights(population_name, generation,
                                                                       (population, weights_size(units))))
        self.dim_state = units[0]
        self.name = "ga"

    #depending on the state the input dim needs to be updated
    def get_state2(self, game, player, food):

        game_matrix = np.zeros(shape=(game.width+2, game.height+2))
        
        for i, coord in enumerate(game.player.position):
            game_matrix[int(coord[1]/game.width), int(coord[0]/game.height)] = 1
        for food in game.food:
            game_matrix[int(food.y_food/game.width), int(food.x_food/game.height)] = 2
        for i in range(game.width+2):
            for j in range(game.height+2):
                if i == 0 or j == 0 or i == game.width+1 or j == game.height+1:
                    game_matrix[i, j] = 1
        return np.asarray(game_matrix).reshape(1, 484)

    #returns the state
    def get_state(self, game, player, food):

        game_matrix = np.zeros(shape=(game.width+2, game.height+2))
        #adds food, snake and border position to a matrix 
        for i, coord in enumerate(game.player.position):
            game_matrix[int(coord[1]/game.width), int(coord[0]/game.height)] = 1
        for food in game.food:
            game_matrix[int(food.y_food/game.width), int(food.x_food/game.height)] = 2
        for i in range(game.width+2):
            for j in range(game.height+2):
                if i == 0 or j == 0 or i == game.width+1 or j == game.height+1:
                    game_matrix[i, j] = 1
        head = player.position[-1]
        player_x, player_y = int(head[0]/game.width), int(head[1]/game.height)
        #boolean vector representing the state
        state = [
            player_x + 1 < game.width+2 and game_matrix[player_y, player_x+1] == 1,  # danger right
            player_x + -1 >= 0 and game_matrix[player_y, player_x-1] == 1,  # danger left
            player_y + -1 >= 0 and game_matrix[player_y-1, player_x] == 1,  # danger up
            player_y + 1 < game.height+2 and game_matrix[player_y+1, player_x] == 1,  # danger down
            player.direction == player.right,
            player.direction == player.left,
            player.direction == player.up,
            player.direction == player.down,
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y  # food down
            ]
        #turns boolean vector into binary one
        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0
        
        return np.asarray(state).reshape(1, self.dim_state)

    def get_weights(self):
        return self.model.get_weights()


# create a keras model given the layers' structure and weights matrix
def create_model_from_units(units, weights):
    if len(units) < 2:
        print("Error: A model has to have at least 2 layers (input and output layer)")
        return None
    model = Sequential()
    added_weights = 0
    layers = len(units)  # considering input layer and first hidden layer are created at the same time
    for i in range(1, layers):
        activation = 'relu'
        if i == layers-1:
            activation = 'softmax'
        if i == 1:
            model.add(Dense(units[i], activation=activation, input_dim=units[0]))
        else:
            model.add(Dense(units[i], activation=activation))
        weight = weights[added_weights:added_weights+units[i-1]*units[i]].reshape(units[i-1], units[i])
        added_weights += units[i-1]*units[i]
        model.layers[-1].set_weights((weight, np.zeros(units[i])))
    return model







# calculating the fitness value by playing a game with the given weights in snake
def cal_pop_fitness(new_population, units, population):
    fitness = []
    deaths = []
    avg_score = []
    max_scores = []
    for i in range(population):
        K.clear_session()
        #weights are taken from either a prev generation or are randomly generated
        weights = new_population[i]
        #model is created with the shape of the NN and the weights
        model = create_model_from_units(units, weights)
        #game runs and stats are stored
        fit, snake_deaths, snake_avg_score, record = run_game2(model)
        snake_avg_score = round(snake_avg_score, 2)
        print('fitness value of snake ' + str(i) + ':  ' + str(fit) +
              '   Deaths: ' + str(snake_deaths) + '   Avg score: ' + str(snake_avg_score) + '   Record: ' + str(record))
        fitness.append(fit)
        deaths.append(snake_deaths)
        avg_score.append(snake_avg_score)
        max_scores.append(record)
    return np.array(fitness), np.array(deaths), np.array(avg_score), np.array(max_scores)


# get the number of weights given a model's layer structure
def weights_size(units):
    s = 0
    for i in range(len(units)-1):
        s += units[i] * units[i+1]
    return s


# get the matrix weigths of the best snake of a specific generation and population
def best_snake_weights(population_name, generation, pop_size):
    path = "./weights/genetic_algorithm/" + str(population_name) + "/stats.txt"
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    last = lines[-1]
    best_snake = last.split(" ")[-2]
    weights_path = "./weights/genetic_algorithm/" + str(population_name) + "/generation_" + str(generation) + ".txt"
    population = np.loadtxt(weights_path)
    population.reshape(pop_size)
    weights = population[int(best_snake)]
    return weights


# use a model to play a game and return model fitness score
def run_game(model):
    # parameters to initialise the game
    steps_per_game = 5000
    max_steps_per_food = 200

    steps = 0
    game = Game(display_option=False)
    game.player=Player(game, "green")
    ga_agent = GAAgent(training=True)  # only used to get the state
    player = game.player
    game.food.append(Food(game))
    food = game.food[0]
    game.game_speed = 0
    player.init_player(game)
    current_step = 0
    slow_penalty = 0
    for _ in tqdm(range(0,steps_per_game),leave=False):
        #gets the state end predicts a move
        state = ga_agent.get_state(game, player, food)
        prediction = model(state)
        move = np.argmax(prediction[0])
        player.do_move(move, game)
        current_step += 1
        #checks if snake has eaten, penalty and crash
        if player.eaten:
            current_step = 0
        if current_step > max_steps_per_food:
            player.crash = True
            slow_penalty += 1
        if player.crash:
            player.init_player(game)
            current_step = 0
        steps += 1
    return player.deaths * (-150) + player.record * 5000 + slow_penalty * (-1000) + int(steps_per_game / (player.total_score + 1)) * (-100), \
           player.deaths, player.total_score / (player.deaths + 1), player.record

#similar to function above but uses different fitness function
def run_game2(model):
    # parameters
    steps_per_game = 5000
    max_steps_per_food = 200

    steps = 0
    game = Game(display_option=False)
    game.player=Player(game, "green")
    ga_agent = GAAgent(training=True)  # only used to get the state
    player = game.player
    game.food.append(Food(game))
    food = game.food[0]
    game.game_speed = 0
    player.init_player(game)
    current_step = 0
    slow_penalty = 0
    previous_states=[]
    for _ in tqdm(range(0,steps_per_game),leave=False):
        state = ga_agent.get_state(game, player, food)
        prediction = model(state)
        move = np.argmax(prediction[0])
        player.do_move(move, game)
        current_step += 1
        if player.eaten:
            current_step = 0
        if  ga_agent.get_state2(game, player, food).tolist() in previous_states:
            previous_states=[]
            player.crash = True
            slow_penalty += 1
        if player.crash:
            player.init_player(game)
            current_step = 0
        steps += 1
        previous_states.append(ga_agent.get_state2(game, player, food).tolist())
    return player.deaths * (-100) + player.record * 8000 + slow_penalty * (-2000), \
           player.deaths, player.total_score / (player.deaths + 1), player.record


# return the history of the training stats as arrays
def get_stats_as_history(population_name):
    path = "weights/genetic_algorithm/" + str(population_name) + "/stats.txt"
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    lines = lines[1::2]
    max_fitness = []
    max_avg_score = []
    avg_fitness = []
    avg_deaths = []
    avg_score = []
    max_score = []
    for line in lines:
        stats = line.split(" ")
        max_fitness.append(int(stats[0]))
        max_avg_score.append(float(stats[1]))
        avg_fitness.append(float(stats[2]))
        avg_deaths.append(float(stats[3]))
        avg_score.append(float(stats[4]))
        max_score.append(int(stats[5]))
    return max_fitness, max_avg_score, avg_fitness, avg_deaths, avg_score, max_score


######################################
##Mutation operators

def geometric_semantic_mutation(offspring_crossover,ms):
    #adds random value between ms and -ms to each genome
    for idx in range(offspring_crossover.shape[0]): #for each offspring
        for i,genome in enumerate(offspring_crossover[idx, :]): #for each gene in the offspring
            random_value = random.uniform(-ms,ms)
            offspring_crossover[idx, i] = genome + random_value
    return offspring_crossover

def standard_mutation(offspring_crossover,p_mutation):
    #changes a value with probability p_mutation into a value between -1 and 1
    for idx in range(offspring_crossover.shape[0]): #for each offspring
        for i,genome in enumerate(offspring_crossover[idx, :]): #for each gene in the offspring
            if random.random()<=p_mutation:
                    offspring_crossover[idx, i]=random.uniform(-1,1) 
    return offspring_crossover

def creep_mutation(offspring_crossover,p_mutation):
    #changes a value with probability p_mutation into a value between min and max value in vector
    for idx in range(offspring_crossover.shape[0]): #for each offspring
        max_val=np.max(offspring_crossover[idx, :])
        min_val=np.min(offspring_crossover[idx, :])
        for i,genome in enumerate(offspring_crossover[idx, :]): #for each gene in the offspring
            if random.random()<=p_mutation:
                    offspring_crossover[idx, i]=random.uniform(min_val,max_val) 
    return offspring_crossover

#Crossover operators
def geometric_semantic_crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    
    for k in range(offspring_size[0]):
        #selects two different parents from the parent pool
        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)
            # produce offspring from two parents if they are different
            if parent1_idx != parent2_idx:
                break
        #calculates new value based on value from parents and random parameter r between 1 and -1
        for i in range(offspring_size[1]):
            r=random.random()
            x=parents[parent1_idx,i]
            y=parents[parent2_idx,i]
            offspring[k,i]=r*x+(1-r)*y
    return offspring


def standard_crossover(parents, offspring_size):
    offspring = []
    
    for _ in range(offspring_size[0]):
        #selects two different parents from the parent pool
        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)
            # produce offspring from two parents if they are different
            if parent1_idx != parent2_idx:
                break
        #selects a point in the array representing the parent
        crossover_point=random.randint(0,offspring_size[1])
        #takes values before crossover point from one parent and the rest from the other
        child=np.hstack([parents[parent1_idx][:crossover_point],parents[parent2_idx][crossover_point:]])
        offspring.append(child)
    return np.array(offspring)

def uniform_crossover(parents, offspring_size):
    offspring = []
    
    for _ in range(offspring_size[0]):
        #selects two different parents from the parent pool
        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)
            # produce offspring from two parents if they are different
            if parent1_idx != parent2_idx:
                break
        child=np.empty(offspring_size[1])
        #takes a genome with probability .5 from each parent
        for i in range(offspring_size[1]):
            if random.random()<=0.5:
                    child[i]=parents[parent1_idx][i]
            else:
                child[i]=parents[parent2_idx][i]
        offspring.append(child)
    return np.array(offspring)



##Selection methods
def fps(pop, fitness, num_parents):
    parents=[]
    #gets the worst snake, this is to handle negative fitness values
    abs_worst_fitness = abs(np.min(fitness)) + 1
    total_score = sum([snake_fitness + abs_worst_fitness for snake_fitness in fitness])
    while len(parents)<num_parents:
        spin = random.uniform(0, total_score)
        position=0
        for fitness_score, parent in zip(fitness,pop):
            updated_score=fitness_score+abs_worst_fitness
            position+=updated_score
            if position>spin:
                parents.append(parent)
                break
    return np.array(parents)

#this assumes that the lower the fitness the better the snake
def fps_min(pop, fitness, num_parents):
    parents=[]
    #gets the worst snake, this is to handle negative fitness values
    abs_worst_fitness = abs(np.min(fitness)) + 1
    #1/total score is used so that snakes with lower fitness have a larger probability of being selected
    total_score = sum([1/(snake_fitness + abs_worst_fitness) for snake_fitness in fitness])
    while len(parents)<num_parents:
        spin = random.uniform(0, total_score)
        position=0
        for fitness_score, parent in zip(fitness,pop):
            updated_score=1/(fitness_score+abs_worst_fitness)
            position+=updated_score
            if position>spin:
                parents.append(parent)
                break
    return np.array(parents)


def ranking(pop, fitness, num_parents):
    selected_snakes = []
    # sort snakes according to their score values
    sorted_snakes = sorted(zip(pop,fitness), key = lambda snake: snake[1], reverse = False)
    # add snakes to the selected_snakes list
    # snakes with a higher ranking have a higher probability of being in the list
    while len(selected_snakes)<num_parents:
        for index, snake in enumerate(sorted_snakes):
            probability = (index + 1) / len(sorted_snakes)
            if random.random() <= probability:
                selected_snakes.append(snake[0])
    
    # shuffles and returns selected snakes in random order
    random.shuffle(selected_snakes)
    return np.array(selected_snakes)

#this assumes that the lower the fitness the better the snake
#only difference to previous function is that the order of the snakes is reversed
def ranking_min(pop, fitness, num_parents):
    selected_snakes = []
    # sort snakes according to their score values
    sorted_snakes = sorted(zip(pop,fitness), key = lambda snake: snake[1], reverse = True)
    # add snakes to the selected_snakes list
    # snakes with a higher ranking have a higher probability of being in the list
    while len(selected_snakes)<num_parents:
        for index, snake in enumerate(sorted_snakes):
            probability = (index + 1) / len(sorted_snakes)
            if random.random() <= probability:
                selected_snakes.append(snake[0])
    
    # shuffles and returns selected snakes in random order
    random.shuffle(selected_snakes)
    return np.array(selected_snakes)

def tournament(pop, fitness, num_parents,tournament_size):
    selected_snakes=[]
    for _ in range(tournament_size):
        snakes_w_fitness=zip(pop,fitness)
        random_sample = random.sample(list(snakes_w_fitness), tournament_size)
        sorted_snakes=sorted(random_sample, key = lambda snake: snake[1], reverse = True)
        selected_snakes.append(sorted_snakes[0][0])
    return np.array(selected_snakes)

#this assumes that the lower the fitness the better the snake
#only difference to previous function is that the order of the snakes is reversed
def tournament_min(pop, fitness, num_parents,tournament_size):
    selected_snakes=[]
    for _ in range(tournament_size):
        snakes_w_fitness=zip(pop,fitness)
        random_sample = random.sample(list(snakes_w_fitness), tournament_size)
        sorted_snakes=sorted(random_sample, key = lambda snake: snake[1], reverse = False)
        selected_snakes.append(sorted_snakes[0][0])
    return np.array(selected_snakes)


######################################




def run_models(mutation,crossover, selection,start_gen=0,maximisation=True):
    units = [12, 32, 4]  # no. of input units, no. of units in hidden layer n, no. of output units
    population = 50  # parameter: population
    num_weights = weights_size(units)  # weights of a single model (snake)
    pop_size = (population, num_weights)  # population size
    #  creating the initial weights
    new_population = np.random.choice(np.arange(-1, 1, step=0.01), size=pop_size, replace=True)
    num_generations =15  # parameter: number of generations
    num_parents_mating = 12  # parameter: number of best parents selected for crossover
    checkpoint = 5  # parameter: how many generations between saving weights
    population_name = mutation+"_"+crossover+"_"+selection  # parameter: name of the population
    current_gen = start_gen  # parameter: last finished generation

    # restore weights from previous generation
    restore_weights_from_txt = start_gen!=0
    if restore_weights_from_txt:
        path = "./Weights_initial_testing/genetic_algorithm/" + str(population_name) + "/generation_" + str(current_gen) + ".txt"
        new_population = np.loadtxt(path)

    # start training
    for generation in tqdm(range(num_generations)):
        # skip old generations
        if restore_weights_from_txt and generation <= current_gen:
            continue
        print('GENERATION ' + str(generation))
        # measuring the fitness of each snake in the population
        fitness, deaths, avg_score, max_scores = cal_pop_fitness(new_population, units, population)
        # print generation stats
        print('fittest snake in geneneration ' + str(generation) + ' : ', np.max(fitness))
        print('highest average score in geneneration ' + str(generation) + ' : ', np.max(avg_score))
        print('average fitness value in geneneration ' + str(generation) + ' : ', np.sum(fitness) / population)
        print('average deaths in geneneration ' + str(generation) + ' : ', np.sum(deaths) / population)
        print('average score in geneneration ' + str(generation) + ' : ', np.sum(avg_score) / population)
        print('max score in geneneration ' + str(generation) + ' : ', max_scores[np.argmax(max_scores)])

        # selecting the best parents in the population for mating
        if maximisation:
            if selection=="tournament":
                parents = tournament(new_population, fitness, num_parents_mating,15)
            elif selection=="fps":
                parents = fps(new_population, fitness, num_parents_mating)
            elif selection=="ranking":
                parents = ranking(new_population, fitness, num_parents_mating)
        else:
            #inverse of fitness so that it becomes a minimisation problem
            fitness_min=list(map(lambda x:1/x,fitness))
            if selection=="tournament":
                parents = tournament_min(new_population, fitness_min, num_parents_mating,15)
            elif selection=="fps":
                parents = fps_min(new_population, fitness_min, num_parents_mating)
            elif selection=="ranking":
                parents = ranking_min(new_population, fitness_min, num_parents_mating)

        # generating next generation using crossover
        if crossover=="uniform":
            offspring_crossover = uniform_crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))
        elif crossover=="standard":
            offspring_crossover = standard_crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))
        elif crossover=="geometric":
            offspring_crossover = geometric_semantic_crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))
        #pop_size[0] - parents.shape[0]
        # adding some variations to the offspring using mutation.
        if mutation=="creep":
            offspring_mutation = creep_mutation(offspring_crossover, 0.1)   
        elif mutation=="standard":
            offspring_mutation = standard_mutation(offspring_crossover, 0.05)  
        elif mutation=="geometric":
            offspring_mutation = geometric_semantic_mutation(offspring_crossover, 0.1)  

        # creating the new population based on the parents and offspring
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
        # new_population = offspring_mutation
        # save generation stats
        dir_path = "weights/genetic_algorithm/" + str(population_name) + "/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path = "weights/genetic_algorithm/" + str(population_name) + "/stats.txt"
        f = open(path, "a+")
        f.write(str(generation) + "\n")
        f.write(str(np.max(fitness)) + " " + str(np.max(avg_score)) + " " + str(np.sum(fitness)/population) + " " +
                str(np.sum(deaths) / population) + " " + str(np.sum(avg_score) / population) + " " +
                str(max_scores[np.argmax(max_scores)]) + " " + str(np.argmax(fitness)) + " \n")
        f.close()

        # save weights matrix
        if generation % checkpoint == 0 or generation == num_generations-1:
            path = "weights/genetic_algorithm/" + str(population_name) + "/generation_" + str(generation) + ".txt"
            np.savetxt(path, new_population)
            print("weights saved")



selected_models=[("creep","standard","tournament"),("creep","uniform","tournament"),("standard","geometric","tournament"),("standard","uniform","tournament"),("creep","geometric","tournament"),("geometric","geometric","tournament")]

for model in selected_models:
    print("#######################################")
    print(model)
    run_models(model[0],model[1],"ranking",maximisation=False)


