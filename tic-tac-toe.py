"""Reinforcement Learning applied to Tic-Tac-Toe

Usage:
    ./tic-tac-toe.py

Author:
    Alexis Flesch - 10/2023

Description:
    * class TicTacToe() is the MDP for the game :
        + a board is a tuple of length 9 where 0 means empty, 1 means player 1 and -1 means player 2
        + player 1 always starts
        + the reward is 1 if player 1 wins, -1 if player 2 wins, 0 if it's a draw and None if the game is not over
        + to know whose turn it is, we count the number of 1 and -1 in the board

    * class QIterations() is an algorithm to approximate the optimal Q-function :
        + Q is a dictionary of dictionaries
        + Q[state][action] is the expected reward of playing action in state
        + The algorithm uses an "epsilon-random" policy to explore the state space :
            - it sometimes plays a random move for its opponent
            - it sometimes plays what it thinks is the best move for its opponent
        + It is not strictly speaking Q-Iteration because it doesn't store the transition probabilities
"""

import numpy as np
from itertools import product
import sys
import json


class TicTacToe():
    """
    Class to represent the Tic-Tac-Toe game
    - stateSpace : list of all possible states (tuples of length 9) :
        * a state can have as many 1 as -1 (if it's player 1's turn) or one more 1 than -1 (if it's player 2's turn)
    - action : index of the cell where the player wants to play (0 to 8)
    """

    def __init__(self):
        self.state_space = self.create_state_space()

    def create_state_space(self):
        """
        Reducing the state space is crucial to save memory and computation time.
        - don't keep boards where one player played twice in a row
        - don't keep boards where both player won
        """
        B = []
        for state in product([-1, 0, 1], repeat=9):
            n1, n2 = state.count(1), state.count(-1)
            if n1 == n2 or n1 == n2+1:
                # Check if both players won
                if self.reward(state, 1) == 1 and self.reward(state, -1) == 1:
                    continue
                B.append(state)
        return B

    def whose_turn(self, state):
        n1, n2 = state.count(1), state.count(-1)
        if n1 == n2:
            return 1
        else:
            return -1

    def action_space(self, state):
        """
        Player can play in any empty cell if the game is not over
        Output: list of cells
        """
        if self.reward(state, 1) is not None:
            return []
        else:
            return [i for i, x in enumerate(state) if x == 0]

    def random_transition(self, state, action, player):
        """
        Updates state :
        - player plays action
        - opponent plays a random move
        - compute reward
        """
        new_state = tuple(player if i ==
                          action else x for i, x in enumerate(state))
        # Check if game is over
        reward = self.reward(new_state, player)
        if reward is not None:
            return new_state, reward

        # If not, opponent plays
        action_space = self.action_space(new_state)
        # Play a random move for the opponent
        opponent_action = np.random.choice(action_space)
        new_state = tuple(-player if i ==
                          opponent_action else x for i, x in enumerate(new_state))
        return new_state, self.reward(new_state, player)

    def smart_transition(self, state, action, player, Q):
        """
        Update state :
        - player plays action
        - opponent plays a move using the Q-function
        """
        new_state = tuple(player if i ==
                          action else x for i, x in enumerate(state))

        # Check if game is over
        reward = self.reward(new_state, player)
        if reward is not None:
            return new_state, reward

        # If not, opponent plays a "good" move according to Q-function
        # That is, the move that minimizes the expected reward of current player
        best_action = max(Q[new_state], key=Q[new_state].get)
        new_state = tuple(-player if i ==
                          best_action else x for i, x in enumerate(new_state))
        return new_state, self.reward(new_state, player)

    def reward(self, new_state, player):
        """
        Compute reward after transition for player
        """
        # Did player win ?
        for i in range(3):
            if new_state[i] == new_state[i+3] == new_state[i+6] == player:
                return 1
            if new_state[3*i] == new_state[3*i+1] == new_state[3*i+2] == player:
                return 1
        if new_state[0] == new_state[4] == new_state[8] == player:
            return 1
        if new_state[2] == new_state[4] == new_state[6] == player:
            return 1
        # Did player lose ?
        for i in range(3):
            if new_state[i] == new_state[i+3] == new_state[i+6] == -player:
                return -1
            if new_state[3*i] == new_state[3*i+1] == new_state[3*i+2] == -player:
                return -1
        if new_state[0] == new_state[4] == new_state[8] == -player:
            return -1
        if new_state[2] == new_state[4] == new_state[6] == -player:
            return -1
        # Is it a draw ?
        if 0 not in new_state:
            return 0
        # Not a draw and not over ?
        return None


class QIterations():
    def __init__(self, MDP=TicTacToe()):
        self.MDP = MDP
        self.Q = dict()
        for s in self.MDP.state_space:
            self.Q[s] = dict()
            for a in self.MDP.action_space(s):
                self.Q[s][a] = 0

    def run(self, N=10, gamma=1, epsilon=.2):
        # apply N iterations of Q-Iteration
        for k in range(N):
            print('Iteration', k+1)
            # Updating the function Q
            for s in self.MDP.state_space:
                # Check whether it's player 1's turn or player 2's turn
                player = self.MDP.whose_turn(s)
                for a in self.MDP.action_space(s):
                    # Epsilon-random policy
                    if np.random.rand() < epsilon:
                        next_state, reward = self.MDP.random_transition(
                            s, a, player)
                    else:
                        next_state, reward = self.MDP.smart_transition(
                            s, a, player, self.Q)
                    # Is the game over ?
                    if reward is not None:
                        # Suppose agent has already lost in this position
                        # but now it doesn't because the opponent played a random move.
                        # In this cas, don't update the Q-function.
                        if self.Q[s][a] != -1:
                            self.Q[s][a] = reward
                        continue
                    # If not, compute the maximum reward expected from next state
                    next_actions = self.MDP.action_space(next_state)
                    maxReward = -np.inf
                    if len(next_actions) > 0:
                        for act in next_actions:
                            maxReward = max(maxReward, self.Q[next_state][act])
                    else:
                        print("This shouldn't happen")
                        maxReward = 0
                    self.Q[s][a] = gamma*maxReward


class QLearning():
    def __init__(self, MDP=TicTacToe()):
        self.MDP = MDP
        self.Q = dict()
        for s in self.MDP.state_space:
            self.Q[s] = dict()
            for a in self.MDP.action_space(s):
                self.Q[s][a] = 0

    def run(self, N=1000, gamma=1, epsilon=.2, alpha=.1):
        # apply N iterations of Q-Iteration
        for k in range(N):
            if k % 10000 == 0:
                print('Iteration', k+1)
            # Choose a random state
            index = np.random.randint(0, len(self.MDP.state_space))
            s = self.MDP.state_space[index]

            # Loop until terminal event
            while self.MDP.reward(s, 1) is None:
                player = self.MDP.whose_turn(s)

                # Epsilon-greedy policy
                if np.random.rand() < epsilon:
                    a = np.random.choice(self.MDP.action_space(s))
                else:
                    a = max(self.Q[s], key=self.Q[s].get)
                next_state, reward = self.MDP.random_transition(s, a, player)

                # Is the game over ?
                if reward is not None:
                    self.Q[s][a] = (1-alpha)*self.Q[s][a] + alpha*reward
                    break

                # If not, compute the maximum reward expected from next state
                next_actions = self.MDP.action_space(next_state)
                maxReward = -np.inf
                for act in next_actions:
                    maxReward = max(maxReward, self.Q[next_state][act])
                # Update Q-function
                self.Q[s][a] = (1-alpha)*self.Q[s][a] + \
                    alpha*(0+gamma*maxReward)

                # Update state
                s = next_state


class QLearning2():
    def __init__(self, MDP=TicTacToe()):
        self.MDP = MDP
        self.Q = dict()
        for s in self.MDP.state_space:
            self.Q[s] = dict()
            for a in self.MDP.action_space(s):
                self.Q[s][a] = 0

    def run(self, N=100, gamma=1, epsilon=.2, alpha=.1, epsilon2=1):
        """
        Parameters:
        - epsilon : for the epsilon-greedy policy
        - epsilon2 : for the epsilon-random policy for the opponent
        """
        # apply N iterations of Q-Iteration
        for k in range(N):
            print('Iteration', k+1)

            # Loop through all states
            for s in self.MDP.state_space:

                # Loop until terminal event
                while self.MDP.reward(s, 1) is None:
                    player = self.MDP.whose_turn(s)

                    # Epsilon-greedy policy
                    if np.random.rand() < epsilon:
                        a = np.random.choice(self.MDP.action_space(s))
                    else:
                        a = max(self.Q[s], key=self.Q[s].get)

                    # Epsilon-random policy for the opponent
                    if np.random.rand() < epsilon2:
                        next_state, reward = self.MDP.random_transition(
                            s, a, player)
                    else:
                        next_state, reward = self.MDP.smart_transition(
                            s, a, player, self.Q)

                    # Is the game over ?
                    if reward is not None:
                        self.Q[s][a] = (1-alpha)*self.Q[s][a] + alpha*reward
                        break

                    # If not, compute the maximum reward expected from next state
                    next_actions = self.MDP.action_space(next_state)
                    maxReward = -np.inf
                    for act in next_actions:
                        maxReward = max(maxReward, self.Q[next_state][act])
                    # Update Q-function
                    self.Q[s][a] = (1-alpha)*self.Q[s][a] + \
                        alpha*(0+gamma*maxReward)

                    # Update state
                    s = next_state


def print_state(state):
    for i in range(3):
        print(state[3*i:3*i+3])
    print()


def convert_dict_to_json(Q):
    """
    Convert the Q dictionary to a JSON-compatible format.
    """
    json_compatible_dict = dict()
    for s, actions in Q.items():
        if s == 'Number of iterations':
            json_compatible_dict[s] = actions
        else:
            s_str = str(s)  # Convert state tuple to a string
            json_compatible_dict[s_str] = dict()
            for a, value in actions.items():
                a_str = str(a)  # Convert action int to a string
                json_compatible_dict[s_str][a_str] = value
    return json_compatible_dict


def load_json_to_dict(json_data):
    """
    Load the JSON data and convert it back to a dictionary of tuples.
    """
    json_dict = json.loads(json_data)
    Q = dict()
    for s_str, actions in json_dict.items():
        if s_str == 'Number of iterations':
            Q[s_str] = actions
        else:
            s = tuple(map(int, s_str.strip('()').split(', ')))
            Q[s] = dict()
            for a_str, value in actions.items():
                a = int(a_str)
                Q[s][a] = value
    return Q


def load_data():
    """
    Load the json data file
    """
    # Check if the file exists
    try:
        # Load the file
        with open('tictactoe.json', 'r') as f:
            json_data = f.read()
            Q = load_json_to_dict(json_data)
            return Q
    except FileNotFoundError:
        return None


def save_data(Q):
    """
    Save the Q-function to a json file
    """
    with open('tictactoe.json', 'w') as f:
        json.dump(convert_dict_to_json(Q), f)


def play_game(Q):
    """
    Toy implementation of the game to test the Q-function.
    Real implementation will be done in javascript.
    """
    AI = 1
    user = -1
    state = (0, 0, 0, 0, 1, 0, 0, 0, 0)
    QI = QIterations()
    while QI.MDP.reward(state, user) is None:
        print_state(state)
        # User plays
        action = int(input('Enter action : '))
        state = tuple(user if i == action else x for i, x in enumerate(state))
        # Computer plays
        if QI.MDP.reward(state, user) is not None:
            break
        # Find best strategy
        best_action = max(Q[state], key=Q[state].get)
        state = tuple(AI if i == best_action else x for i,
                      x in enumerate(state))
    print('Game over !', QI.MDP.reward(state, user))


if __name__ == '__main__':
    # Number of iterations
    # Recall that the number of states is close to 6000
    # So for Q-Learning, an iteration being a random choice of a state,
    # we need a lot of iterations !
    # For Q-Learning2 and Q-iteration, an iteration is a loop through all states
    # Load the Q function
    Q = load_data()
    # Q = None
    QL = QLearning2()
    if Q is not None:
        QL.Q = Q
        iterations = Q['Number of iterations']
    else:
        iterations = 0
    # Train against a random opponent
    QL.run(N=50, epsilon2=1)
    QL.run(N=200, epsilon2=0)
    QL.Q['Number of iterations'] = 250 + iterations
    print("Number of iterations so far :",
          QL.Q['Number of iterations'])
    save_data(QL.Q)
