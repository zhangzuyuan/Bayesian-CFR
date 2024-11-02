import numpy as np
import collections
import torch
import os
import pickle
import copy
from rlcard.utils.utils import *
import random
import multiprocessing
from multiprocessing import Manager, Pool, Process, Queue

class MCCFRagent():
    ''' Implement CFR algorithm
    '''

    def __init__(self, env, m, K=4,isAbs=True, CFR_num=1, tra_num=10, model_path='./cfr_plus_model'):
        ''' Initilize Agent

        Args:
            env (Env): Env class
        '''
        self.use_raw = False
        self.env = env
        self.m = m
        self.K = K
        self.model_path = model_path
        self.isAbs = isAbs
        self.CFR_num = CFR_num
        self.tra_num = tra_num

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict()
        self.average_policy =collections.defaultdict()
        
        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict()
        self.iteration = 0
        self.oppoCV = None

    def train(self):
        import time 
        # costtime = []
        total_start = time.perf_counter()
        for t in range(0, self.CFR_num):
            self.iteration += 1
            # print('iteration is ',self.iteration)
            # start = time.perf_counter()
            typ = random.randint(0, self.m-1)
            for player_id in range(self.env.num_players):
                regretMemory = collections.defaultdict()
                policyMemory = collections.defaultdict()
                for k in range(self.K):
                    self.env.reset(typ)
                    prob = 1
                    self.traverse_tree(max(0, self.iteration - 10), prob, player_id, regretMemory, policyMemory)
                for item in regretMemory:
                    self.feed(item, player_id,isRegret=True)
                for item in policyMemory:
                    self.feed(item, player_id,isRegret=False)
        self.update_policy()
                

            # for player_id in range(self.env.player_num):
            #     # print('now is player', player_id)
            #     multiprocessing.freeze_support()
            #     q = multiprocessing.Manager().Queue()
            #     p = multiprocessing.Pool(Process_num)
            #     j = Process_num
            #     #orch.multiprocessing.spawn(fn=self.multi_traverse, args = (q, player_id, self.tra_num), nprocs=Process_num, join=True, daemon=False)
            #     for k in range(Process_num):
            #         p.apply_async(self.multi_traverse, args = (q, player_id, self.tra_num))
            #     p.close()
            #     p.join()
            #     for k in range(Process_num):
            #         u = q.get()
            #         r = u[0]
            #         pl = u[1]
            #         self.feed(r, player_id, isRegret=True)
            #         self.feed(pl, player_id, isRegret=False)
            # end = time.perf_counter()
            # print('iteration {} cost {}s'.format(self.iteration, end - start))
            # costtime.append(end - start)

        # total_end = time.perf_counter()
        # print("epoch {} cost {}s, average is {}s".format(i+1, total_end - total_start, np.mean(costtime)))

    def feed(self, dicts, player_id, isRegret=True):
        if isRegret:
            for obs in dicts:
                regrets = dicts[obs]
                if obs not in self.regrets:
                    self.regrets[obs] = np.zeros(self.env.num_actions)
                for action in range(self.env.num_actions):
                    self.regrets[obs][action] = max(0, self.regrets[obs][action] + regrets[action])
                self.regret_matching(obs)
        else:
            for obs in dicts:
                probs = dicts[obs]
                for action in range(self.env.num_actions):
                    self.average_policy[obs] = np.zeros(self.env.num_actions)
                self.average_policy[obs] = self.average_policy[obs] + probs

    def multi_traverse(self, q, player_id, num):
        regretMemory = collections.defaultdict()
        policyMemory = collections.defaultdict()
        for i in range(num):
            self.env.init_game()
            prob = 1
            self.traverse_tree(max(0, self.iteration - 10), prob, player_id, regretMemory, policyMemory)
        q.put([regretMemory, policyMemory])
        return [regretMemory, policyMemory]


    def traverse_tree(self, w, prob, player_id, regretMemory, policyMemory):    
        try:
            if self.env.is_over():
                return self.env.get_payoffs()

            try:
                current_player = self.env.get_player_id()
                action_utilities = {}
                state_utility = np.zeros(self.env.num_players)
                obs, legal_actions,_ = self.get_state(current_player)
                action_probs = self.action_probs(obs, legal_actions, self.policy)
            except Exception as e:
                print(str(e))

            if current_player == player_id:
                try:
                    for action in legal_actions:
                        action_prob = action_probs[action]
                        self.env.step(action)
                        utility = self.traverse_tree(w, prob, player_id, regretMemory, policyMemory)
                        self.env.step_back()

                        state_utility += action_prob * utility
                        action_utilities[action] = utility[player_id]

                    if obs not in regretMemory:
                        regretMemory[obs] = np.zeros(self.env.num_actions)

                    for action in legal_actions:
                        regretMemory[obs][action] = (action_utilities[action] - state_utility[current_player])
                except Exception as e:
                    print(str(e))

            else:
                try:
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                    action_prob = action_probs[action]
                    prob_temp = prob * action_prob
                    self.env.step(action)
                    state_utility = self.traverse_tree(w, prob_temp, player_id, regretMemory, policyMemory)
                    self.env.step_back()

                    if obs not in policyMemory:
                        policyMemory[obs] = np.zeros(self.env.num_actions)

                    for action in legal_actions: 
                        policyMemory[obs][action] = w * prob * action_prob
                except Exception as e:
                    print(str(e))

            return state_utility
        except Exception as e:
            print(str(e))

    def update_policy(self):
        ''' Update policy based on the current regrets
        '''
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def compute_exploitability_mlprocess(self, eval_num, process_num):
        from multiprocessing import Manager
        multiprocessing.freeze_support()
        u = []
        p = Pool(process_num)
        for i in range(0, process_num):
            u.append(p.apply_async(self.compute_exploitability, args=(int(eval_num/process_num),)))
        p.close()
        p.join()

        for i in range(0, len(u)):
            u[i] = u[i].get()

        return np.mean(u)

    def regret_matching(self, obs):
        ''' Apply regret matching

        Args:
            obs (string): The state_str
        '''
        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)

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

    def set_evalenv(self, eval_env):
        self.eval_env = eval_env

    def get_evalenv(self):
        return self.eval_env

    def get_policy(self, obs, legal_actions, isEval=False, isSubgame=False, oppoCV = None):
        # obs = self.state_to_obs(state)
        # obs = state['obs'].tostring()
        if isEval and not isSubgame:
            return self.action_probs(obs, legal_actions, self.average_policy)
        if isEval:
            '''
            if state['round'] >= 1:
                import agents.subgame_resolving as subgame
                Subgame = subgame.subgame(self.eval_env, self, state['current_player'], oppoCV)
                policy, self.oppoCV = Subgame.resolve() 
                return self.action_probs(obs.tostring(), legal_actions, policy)
            else:
            '''
            return self.action_probs(obs, legal_actions, self.average_policy)
        else:
            return self.action_probs(obs, legal_actions, self.policy)

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
        try:
            if obs not in policy:
                action_probs = np.array([1.0/self.env.num_actions for _ in range(self.env.num_actions)])
                self.policy[obs] = action_probs
            else:
                action_probs = policy[obs]
            action_probs = remove_illegal(action_probs, legal_actions)
        except Exception as e:
            print(str(e))
        # print
        return action_probs

    def eval_step(self, state, env=None):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
        '''
        probs = self.get_policy(state['obs'].tostring(), list(state['legal_actions'].keys()), isEval=True, oppoCV=self.oppoCV)
        action = np.random.choice(len(probs), p=probs)
        info = {}
        return action, info

    def exp_step(self, state, env=None):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
        '''
        probs = self.get_policy(state['obs'].tostring(), list(state['legal_actions'].keys()), isEval=True, oppoCV=self.oppoCV)
        # action = np.random.choice(len(probs), p=probs)
        info = {}
        return probs.tolist(), info

    def compute_exploitability(self, eval_num):
        self.cards = init_standard_deck()
        self.env.init_game()
        self.set_evalenv(self.env)
        palyers_range=[]
        Range = []

        for i in range(0,52):
            for j in range(i+1,52):
                Range.append([[self.cards[i], self.cards[j]], 1/1326])

        for player_id in range(self.env.num_players):
            palyers_range.append(Range)
        
        utility = []
        for i in range(0, eval_num):
            for player_id in range(self.env.num_players):
                self.t = 0
                self.env.init_game()
                self.oppoCV_temp = []
                self.oppoCV_temp2 = []
                self.oppoCV_o = None
                for i in range(0,len(Range)):
                    self.oppoCV_temp.append(None)
                    self.oppoCV_temp2.append(None)
                #print(palyers_range[0][0:3])
                palyers_range_ = copy.deepcopy(palyers_range)
                self.oppo_last_action = -1
                payoffs = self.traverse_exp(player_id, palyers_range_)
                #print("final",palyers_range[0][0:3])
                #print('player:{},payoff:{}'.format(player_id,payoffs[player_id]))
                #exit()
                utility.append(payoffs[player_id])
        
        return np.mean(utility)

    def traverse_exp(self, player_id, palyers_range):
        self.t=self.t+1
        if self.env.is_over():
            #print(self.env.get_payoffs())
            return self.env.get_payoffs()
        temp = 0
        current_player = self.env.get_player_id()
        _, legal_actions, state = self.get_state(current_player)
        #print('current_player:{}, legal_actions:{}'.format(current_player, legal_actions))
        #print("in",palyers_range[current_player][0:3])
        if current_player != player_id:
            action_probs = self.get_policy(state, legal_actions, isEval=True, isSubgame=True, oppoCV=self.oppoCV_o)
            self.oppoCV_o = self.oppoCV
            action = np.random.choice(len(action_probs), p=action_probs)
            self.oppo_last_action = action
            self.oppo_last_legal_actions = legal_actions
            self.oppo_last_state = state
            self.last_eval_env = copy.deepcopy(self.eval_env)
            self.env.step(action)

        else:
            #更新对手range
            self.temp_eval_env = self.get_evalenv()
            Sum = 0
            handcards = state['hand']
            publiccards = state['public_cards']
            for i,obj in enumerate(palyers_range[1-current_player]):
                oppocards = [x.get_index() for x in obj[0]]
                if len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)):
                    palyers_range[1-current_player][i][-1] = 0
                    temp = i
                else:
                    if self.oppo_last_action == -1:
                        action_prob = 1
                    else:
                        self.set_evalenv(self.last_eval_env)
                        oppo_state = self.oppo_last_state
                        oppo_state['hand'] = oppocards
                        self.eval_env.game.players[1 - current_player].hand = obj[0]
                        action_prob_ = self.get_policy(oppo_state, self.oppo_last_legal_actions, isEval=True, isSubgame=True, oppoCV=self.oppoCV_temp2[i])
                        action_prob = action_prob_[self.oppo_last_action]
                        self.oppoCV_temp2[i] = self.oppoCV
                    palyers_range[1-current_player][i][-1] = palyers_range[1-current_player][i][-1]*action_prob
                    Sum = Sum + palyers_range[1-current_player][i][-1]

            self.set_evalenv(self.temp_eval_env)
            for i,obj in enumerate(palyers_range[1-current_player]):
                palyers_range[1-current_player][i][-1] = palyers_range[1-current_player][i][-1]/Sum

            action = self.LocalBR(player_id, state, legal_actions, palyers_range[1-current_player])#对两人情况
            self.env.step(action)
        
        #print("out",palyers_range[current_player][0:3],"---",current_player, palyers_range[current_player][temp])
        #print("times:{},palyer:{},action:{}".format(self.t,current_player,action))
        return self.traverse_exp(player_id, palyers_range)

    def LocalBR(self, player_id, state, legal_actions, oppo_range):
        values = np.zeros(self.env.num_actions)
        handcards = state['hand']
        publiccards = state['public_cards']
        pot_myself = state['all_chips'][0]
        pot_oppo = state['all_chips'][1]
        self.temp_eval_env = copy.deepcopy(self.get_evalenv())

        wp = self.WpRollout(player_id, handcards, publiccards, oppo_range)
        asked = pot_oppo - pot_myself 
        #print('asked',asked)
        values[0] = wp*pot_myself-(1-wp)*asked
        for action in legal_actions:
            if action >=2 :
                fp = 0
                oppo_range_temp = copy.deepcopy(oppo_range)
                Sum = 0
                self.env.step(action)
                _, oppo_legal_actions,state_ = self.get_state(1-player_id)
                #print(action, oppo_legal_actions)
                diff = state_['all_chips'][1] - pot_myself
                oppo_state = state_
                #print(state_['all_chips'],pot_myself+action-2)
                for i,obj in enumerate(oppo_range_temp):
                    oppocards = [x.get_index() for x in obj[0]]
                    prob = obj[-1]
                    if len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)):
                        oppo_range_temp[i][-1] = 0
                    else:
                        oppo_state['hand'] = oppocards
                        self.eval_env = copy.deepcopy(self.env)
                        self.eval_env.game.players[1 - player_id].hand = obj[0]
                        foldprob = self.get_policy(oppo_state, oppo_legal_actions, isEval=True, isSubgame=True, oppoCV=self.oppoCV_temp[i])[1]
                        self.oppoCV_temp[i] = self.oppoCV
                        fp = fp + prob*foldprob
                        oppo_range_temp[i][-1] = oppo_range_temp[i][-1]*(1-foldprob)
                        Sum = Sum + oppo_range_temp[i][-1]
                
                if Sum==0:
                    Sum = 1
                for i,obj in enumerate(oppo_range_temp):
                    oppo_range_temp[i][-1] = oppo_range_temp[i][-1]/Sum
                
                self.env.step_back()

                wp = self.WpRollout(player_id, handcards, publiccards, oppo_range_temp)
                values[action] = fp*pot_myself + (1-fp)*(wp*(pot_myself+diff)-(1-wp)*(asked+diff))
                #print("fp",fp," Sum",Sum,'value',values[action])
                #print("action:",action,'--',values[action])
        result = np.argmax(values)
        #print(values)
        self.set_evalenv(self.temp_eval_env)
        if values[result]>0:
            return result
        else:
            return 1#flod

    def WpRollout(self, player_id, handcards, publiccards, oppo_range):
        from rlcard.games.limitholdem.utils import compare_hands
        wp = 0 
        handcards = handcards
        publiccards = publiccards
        for i,obj in enumerate(oppo_range):
            oppocards = [x.get_index() for x in obj[0]]
            prob = obj[-1]
            if prob!=0:
                if len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)):
                    pass
                else:
                    cards = self.cards
                    #print('enter')
                    if len(publiccards)<5:
                        cards = [i.get_index() for i in cards if (i.get_index() not in handcards and i.get_index() not in oppocards and i.get_index() not in publiccards)]
                        for i in range(20):
                            publiccards_temp = list(np.random.choice(a=cards, size=5-len(publiccards)))+publiccards
                            selfcards_temp = publiccards_temp.extend(handcards) 
                            oppocards_temp = publiccards_temp.extend(oppocards) 
                            wp = wp + 0.05*prob*compare_hands(selfcards_temp, oppocards_temp)[0]
                    else:
                        publiccards_temp = publiccards
                        selfcards_temp = publiccards_temp.extend(handcards) 
                        oppocards_temp = publiccards_temp.extend(oppocards) 
                        wp = wp + prob*compare_hands(selfcards_temp, oppocards_temp)[0]
        return wp

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
        obs = self.state_to_obs(state)
        return obs.tostring(), state['legal_actions'],state

    def state_to_obs(self, state):
        public_cards = state['public_cards']
        hand = state['hand']
        idx = [self.env.card2index[card] for card in public_cards]
        obs = np.zeros(54)
        obs[idx] = 1
        idx = [self.env.card2index[card] for card in hand]
        obs[idx] = 1
        if self.isAbs == True:
            if state['all_chips'][0]>state['all_chips'][1]:
                obs[6] = 1
            else:
                obs[7] = 1
        else:
            obs[6] = state['all_chips'][0]
            obs[7] = state['all_chips'][1]
        return obs

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

