import copy
import random
import numpy as np
import collections

import os
import pickle

from rlcard.utils.utils import *

class CFRAgent():
    ''' Implement CFR (chance sampling) algorithm
    '''

    def __init__(self, env, m,model_path='./cfr_model'):
        ''' Initilize Agent

        Args:
            env (Env): Env class
        '''
        self.use_raw = False
        self.env = env
        self.model_path = model_path
        self.m = m

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)

        self.iteration = 0

    def train(self):
        ''' Do one iteration of CFR
        '''
        self.iteration += 1
        # Firstly, traverse tree to compute counterfactual regret for each player
        # The regrets are recorded in traversal
        typ = random.randint(0, self.m-1)
        print(typ)
        # if typ == 2:
        #     print('train')
        for player_id in range(self.env.num_players):
            self.env.reset(typ)
            probs = np.ones(self.env.num_players)
            self.traverse_tree(probs, player_id)

        # Update policy
        self.update_policy()

    def traverse_tree(self, probs, player_id):
        ''' Traverse the game tree, update the regrets

        Args:
            probs: The reach probability of the current node
            player_id: The player to update the value

        Returns:
            state_utilities (list): The expected utilities for all the players
        '''
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()

        action_utilities = {}
        state_utility = np.zeros(self.env.num_players)
        obs, legal_actions,state = self.get_state(current_player)
        action_probs = self.action_probs(obs, legal_actions, self.policy)

        for action in legal_actions:
            action_prob = action_probs[action]
            new_probs = probs.copy()
            new_probs[current_player] *= action_prob

            # Keep traversing the child state
            self.env.step(action)
            utility = self.traverse_tree(new_probs, player_id)
            self.env.step_back()

            state_utility += action_prob * utility
            action_utilities[action] = utility

        if not current_player == player_id:
            return state_utility

        # If it is current player, we record the policy and compute regret
        player_prob = probs[current_player]
        counterfactual_prob = (np.prod(probs[:current_player]) *
                                np.prod(probs[current_player + 1:]))
        player_state_utility = state_utility[current_player]

        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)
        for action in legal_actions:
            action_prob = action_probs[action]
            regret = counterfactual_prob * (action_utilities[action][current_player]
                    - player_state_utility)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += self.iteration * player_prob * action_prob
        return state_utility

    def update_policy(self):
        ''' Update policy based on the current regrets
        '''
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def regret_matching(self, obs):
        ''' Apply regret matching

        Args:
            obs (string): The state_str
        '''
        regret = self.regrets[obs]
        positive_regret_sum = sum([r for r in regret if r > 0])

        action_probs = np.zeros(self.env.num_actions)
        if positive_regret_sum > 0:
            for action in range(self.env.num_actions):
                action_probs[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            for action in range(self.env.num_actions):
                action_probs[action] = 1.0 / self.env.num_actions
        return action_probs

    def action_probs(self, obs, legal_actions, policy):
        ''' Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        '''
        # print(policy)
        if obs not in policy.keys():
            action_probs = np.array([1.0/self.env.num_actions for _ in range(self.env.num_actions)])
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        # print(action_probs)
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs
    
    def get_policy(self, obs, legal_actions, isEval=False):
        if isEval:
            return self.action_probs(obs, legal_actions, self.average_policy)
        else:
            return self.action_probs(obs, legal_actions, self.policy)

    def eval_step(self, state):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
            info (dict): A dictionary containing information
        '''
        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.average_policy)
        action = np.random.choice(len(probs), p=probs)

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        # print(action)

        return action, info
    
    def exp_step(self, state):
        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.average_policy)
        # action = np.random.choice(len(probs), p=probs)

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return probs.tolist(), info



    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        # print(type(state['obs']))
        return state['obs'].tostring(), list(state['legal_actions'].keys()), state 
    

    def compute_exploitability(self, eval_num):
        self.cards = [c.get_index() for c in init_standard_deck()]
        self.env.reset()
        palyers_range=[]
        Range = []

        for i in range(0,52):
            for j in range(i+1,52):
                Range.append([self.cards[i], self.cards[j],1/1326])
        
        for player_id in range(self.env.num_players):
            palyers_range.append(Range)
        
        utility = []
        for i in range(eval_num):
            for player_id in range(self.env.num_players):
                self.t = 0
                self.env.reset()
                palyers_range_ = copy.deepcopy(palyers_range)
                self.oppo_last_action = -1
                payoffs = self.traverse_exp(player_id, palyers_range_)
                utility.append(payoffs[player_id]*2)
        
        return np.mean(utility)
    
    def traverse_exp(self, player_id, palyers_range):
        self.t = self.t + 1
        if self.env.is_over():
            return self.env.get_payoffs()
        
        temp = 0
        current_player = self.env.get_player_id()
        obs, legal_actions, state = self.get_state(current_player)

        if current_player != player_id:
            action_probs = self.get_policy(obs, legal_actions, isEval=True)
            action = np.random.choice(len(action_probs), p=action_probs)
            self.oppo_last_action = action
            self.oppo_last_legal_actions = legal_actions
            # self.oppo_last_chips = state['all_chips']
            self.env.step(action)


    def save(self):
        ''' Save model
        '''
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'wb')
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'wb')
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        ''' Load model
        '''
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'rb')
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'rb')
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()

