import random

import numpy as np

from rlcard.utils.utils import set_seed
import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import norm
class reward():
    def __init__(self):
        pass

    def computer_reward(self, agent1, agent2, evaluate_num, eval_env):
        eval_env.set_agents([agent1, agent2])
        try:
            agent1.set_evalenv(eval_env)
        except:
            pass
        try:
            agent2.set_evalenv(eval_env)
        except:
            pass

        # u = []
        # p = Pool(process_num)
        # for i in range(process_num):
        #     u.append(p.apply_async(self.traverse, args=(agent1, agent2, int(evaluate_num/process_num), eval_env)))

        # for i in range(0, len(u)):
        #     u[i] = u[i].get()
        u = self.traverse(agent1, agent2, evaluate_num, eval_env)
   
        return u

    def traverse(self, agent1, agent2, evaluate_num, eval_env):
        reward = []
        set_seed(random.randint(0,100))
        for eval_episode in range(evaluate_num):
            try:
                agent1.oppoCV = None
            except:
                pass
            try:
                agent2.oppoCV = None
            except:
                pass
            his, payoffs = eval_env.run(is_training=False)
            reward.append(payoffs[0])
        #print(reward)
        return np.mean(reward)

class exploitability():
    def __init__(self):
        pass

    def computer_exploitability(self, agent, evaluate_num):
        from multiprocessing import Manager
        self.agent = agent
        # multiprocessing.freeze_support()
        # u = []
        # p = Pool(process_num)
        # for i in range(process_num):
        #     u.append(p.apply_async(self.traverse, args=(int(evaluate_num/process_num),)))

        # p.close()
        # p.join()

        # for i in range(0, len(u)):
        #     u[i] = u[i].get()
        u = self.traverse(evaluate_num)


        return u

    def traverse(self, evaluate_num):
        reward = []
        set_seed(random.randint(0,100))
        reward.append(self.agent.compute_exploitability(evaluate_num))

        return np.mean(reward)
    
# 计算余弦相似度
def calculate_cosine_similarity(vec1, vec2):
    # 保证两向量非零
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0001
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# 计算似然值为余弦相似度
def calculate_likelihood(observed_vec, data_vec):
    return calculate_cosine_similarity(observed_vec, data_vec)