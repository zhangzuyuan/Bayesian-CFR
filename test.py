''' An example of solve Leduc Hold'em with CFR (chance sampling)
'''
import os
import argparse

import numpy as np
import utils
import rlcard
from rlcard.utils import exploitability
from rlcard.agents import (
    CFRAgent,
    RandomAgent,
)
from rlcard.agents.deep_cfr_agent import DeepCFRAgent
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)
# r = utils.exploitability()
def train(args):
    # Make environments, CFR only supports Leduc Holdem
    env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': 0,
        }
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Initilize CFR Agent
    agent = CFRAgent(
        env,
        os.path.join(
            args.log_dir,
            'cfr_model',
        ),
    )
    # agent.load()  # If we have saved model, we first load the model
    # agent = DeepCFRAgent(
    #     env)
    agent1 = RandomAgent(num_actions=env.num_actions)

    # Evaluate CFR against random
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    data = {}
    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            agent.train()
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            if episode % args.evaluate_every == 0:
                # agent.save() # Save model
                # e1 = np.mean(r.computer_exploitability(agent,50))
                # Generate data from the environment
                e = exploitability(env, agent, 50)
                # prins
                trajectories, payoffs = eval_env.run(is_training=False, typ=1)

                # print(len(trajectories))
                obs = b''
                for i in range(len(trajectories)):
                    for j in range(len(trajectories[i])):
                        if type(trajectories[i][j])!=dict:
                            obs+= str(trajectories[i][j]).encode('utf-8')
                        else:
                            obs+=trajectories[i][j]['obs'].tostring()
                                
                if obs not in data.keys():
                    data[obs] = 0
                data[obs]+=1
                print(data)

                # Reorganaize the data to be state, action, reward, next_state, done
                # trajectories = reorganize(trajectories, payoffs)
                logger.log_performance(
                    episode,
                    tournament(
                        eval_env,
                        args.num_eval_games
                    )[0]
                    # e1
                )
                # print(e1)

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'cfr')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_cfr_result/',
    )

    args = parser.parse_args()

    train(args)
    