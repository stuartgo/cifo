import random
import numpy as np
import pandas as pd

class Snake:
    """
    A class used to represent a snake.

    ...

    Attributes
    ----------
    size_game: int
        size of the game board (e.g., 15)
    init: tuple
        initial position of the snake (e.g., (4, 6))
        assigned randomly if left unspecified
    representation

    optim: str
        
    """

    def __init__(
        self,
        size_game,
        decision_model,
        init_x = None,
        init_y = None,
        weights = None
    ):

        # assign random snake spawn location if initial location is not specified
        if (init_x is None) and (init_y is None):
            self.pos = [(
                random.randint(0, size_game-1),
                random.randint(0, size_game-1)
            )]
        # spawn snake at the specified location if it is inside the screen
        elif (init_x >= 0) and (init_x < size_game) and (init_y >= 0) and (init_y < size_game):
            self.pos = [(init_x, init_y)]
        # raise exception if specified spawning location is off screen
        else:
            raise Exception("init_x and init_y must be integers in the interval [0, size_game-1]")

        # initialize snake attributes
        self.len = len(self.pos)
        self.direction = "up"
        self.score = 0
        self.game_info = None
        self.decision_model = decision_model
        self.size_game = size_game

        # if weights passed as argument, assign each weight to a layer of the neural net
        if weights is not None:
            for index, layer_weights in enumerate(weights):
                self.decision_model.layers[index].set_weights(layer_weights)
        # get weights of the layers
        if self.decision_model=="random":
            self.weights=None
        else:
            self.weights = self.get_weights()


    def get_weights(self):
        """
        Retrieves weights of the layers in the neural network.
        """

        weights = []

        for layer in self.decision_model.layers:
            weights.append(layer.get_weights())
            
        return weights


    def set_score(self, score):
        """
        Update the score of the current snake.
        """

        self.score = score


    def get_score(self):
        """
        Returns the score of the snake in the current game.

        Returns
        -------
        score: int
            total score of the snake
        """

        return self.score


    def move(self, state):
        """
        Updates the position of the snake's head and body
        according to the direction it's moving in.

        Returns
        -------
        keystroke: str or None
            direction in which the snake is moving if turning
            None if the snake's direction is unchanged

        """

        # determine direction in which to move based on the output of the decision model
        if self.decision_model=="random":
            keystroke=random.choice([None, "left", "right"])
        else:
            predictions = self.decision_model(np.array([np.array(state)]))[0]
            index = np.where(predictions == np.amax(predictions))[0][0]
            keystroke = [None, "left", "right"][index]

        # store current position of the head
        current_pos = self.pos[0]
        # remove last position on the list (snake's tail)
        self.pos.pop()
        # update position of the snake based on moving direction
        if (
            (keystroke == "left" and self.direction == "up")
            or (keystroke == "right" and self.direction == "down")
            or (self.direction == "left" and keystroke == None)
        ):
            self.direction = "left"
            self.pos.insert(0, (current_pos[0]-1, current_pos[1]))
            
        elif (
            (keystroke == "left" and self.direction == "down")
            or (keystroke == "right" and self.direction == "up")
            or (self.direction == "right" and keystroke == None)
        ):
            self.direction = "right"
            self.pos.insert(0, (current_pos[0]+1, current_pos[1]))
        elif (
            (keystroke == "left" and self.direction == "right")
            or (keystroke == "right" and self.direction == "left")
            or (self.direction == "up" and keystroke == None)
        ):
            self.direction = "up"
            self.pos.insert(0, (current_pos[0], current_pos[1]+1))
        elif (
            (keystroke == "left" and self.direction == "left")
            or (keystroke == "right" and self.direction == "right")
            or (self.direction == "down" and keystroke == None)
        ):
            self.direction = "down"
            self.pos.insert(0, (current_pos[0], current_pos[1]-1))
        return keystroke
        




class Game:
    """
    A class to represent each game.
    
    ...

    Attributes
    ----------
    size_game: int
        size of the game board (e.g., 20) 
    snake: ????????
        ?????????????????????

    """

    def __init__(self, size_game, snake):
        self.snake = snake
        self.apple = (
                random.randint(0, size_game-1),
                random.randint(0, size_game-1)
            )
        self.size_game = size_game
        self.distance_apple = self.calc_distance()


    def move_apple(self):
        self.apple = (
                random.randint(0, self.size_game-1),
                random.randint(0, self.size_game-1)
            )

    def check_loss(self):
        """
        Checks for game losing conditions.

        """

        # check if snake has hit itself
        if len(self.snake.pos)>1:
            if self.snake.pos[0] in self.snake.pos[1:]:
                return True
        # check if snake has hit a wall (game border)
        if (
            (self.size_game <= self.snake.pos[0][0])
            or (self.size_game <= self.snake.pos[0][1])
            or (self.snake.pos[0][0] < 0)
            or (self.snake.pos[0][1] < 0)
        ):
            return True
        
        return False


    def check_apple(self):
        """
        Checks if the apple is at the same position
        as the snake's head.
        """
        
        # increases score by 20 if snake eats an apple and changes its location
        if self.snake.pos[0] == self.apple:
            self.snake.set_score(self.snake.get_score() + 20)
            self.move_apple()


    def calc_distance(self):
        """
        Determines the distance between the position of
        the snake's head and the currently available apple.

        Returns
        -------
        distance: int
            manhattan distance between the snake's head and the apple
        """

        distance = abs(self.snake.pos[0][0] - self.apple[0]) + abs(self.snake.pos[0][1]-self.apple[1])

        return distance


    def distance_score(self):
        """
        Updates the score of the snake with each movement.
        """

        new_distance = self.calc_distance()
        
        # increments score if the snake is moving closer to the apple
        if new_distance < self.distance_apple:
            self.snake.set_score(self.snake.get_score()+1)
        # decreases score if snake is moving farther away from the apple
        else:
            self.snake.set_score(self.snake.get_score()-2)
        self.distance_apple = new_distance


    def get_state(self):
        """
        Returns the state of the ongoing game.

        Returns
        -------
        state: list of lists
            0 for all points on map, 1 where snake is present and 2 where apple is??????
        """

        state = np.zeros((self.size_game,self.size_game))

        for snake_point in self.snake.pos:
            state[snake_point[0]][snake_point[1]] = 1
        apple = self.apple
        state[apple[0]][apple[1]] = 2

        return state


    def run(self):
        """
        Run a game until the snake fails.

        Returns
        -------
        game_info: list of tuples, tuple, str
            list of tuples: position of all the blocks that make up the snake
            tuple: position of the game's last apple
            str: final choice made by the snake
        """

        game_info = []
        previous_states=[]
        while True:
            if self.check_loss():
                self.snake.set_score(self.snake.get_score() - 100)
                lost=True
                break
            state=self.get_state()
            for prev_state in previous_states:
                if np.array_equal(state,prev_state):
                    self.snake.set_score(self.snake.get_score() - 1000)
                    return game_info
            previous_states.append(state)
            decision = self.snake.move(state = state)

            self.distance_score()
            self.check_apple()
            game_info.append((self.snake.pos.copy(), self.apple, decision))
            self.snake.game_info = game_info
            

        return game_info



# run this to save to file that can then be visualized
# snakey=Snake(20,"random")
# gamey=Game(100,snakey)
# info=gamey.run()
# results=pd.DataFrame(info)
# results.columns=["Pos. snake","Pos. apple","Decision"]
# results.to_csv("Random snake.csv")
# print("done")