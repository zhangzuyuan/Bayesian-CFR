


import os
import random
from arguments import get_args
import rlcard
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.utils import exploitability
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)
from torch.utils.tensorboard import SummaryWriter

from rlcard.utils.utils import get_device, reorganize

def train(args):
    device = get_device()
    writer = SummaryWriter(log_dir = args.log_dir)
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )
    set_seed(args.seed)
    # 训练
    if args.algo == 'cfr':
        from rlcard.agents import CFRAgent
        agent = CFRAgent(
            env,
            args.m,
            os.path.join(
                args.log_dir,
                'cfr_model',
            ),
        )
    if args.algo == 'deep_cfr':
        from rlcard.agents.deep_cfr_agent import DeepCFRAgent
        agent = DeepCFRAgent(
            env,
            m=args.m,
        )
    if args.algo == 'bcfr':
        from rlcard.agents.bcfr_agent import BCFRAgent
        agent = BCFRAgent(
            env,
            args.m,
            os.path.join(
                args.log_dir,
                'bcfr_model',
            ),
        )
    if args.algo == 'deep_bcfr':
        from rlcard.agents.deep_bcfr_agent import DeepBCFRAgent
        agent = DeepBCFRAgent(
            env,
            m=args.m,
        )
    if args.algo == 'mccfr':
        from rlcard.agents.mccfr_agent import MCCFRagent
        agent = MCCFRagent(
            env,
            args.m,
        )
    if args.algo == 'cfr_plus':
        from rlcard.agents.cfr_plus_agent import CFRPlusAgent
        agent = CFRPlusAgent(
            env,
            args.m,
        )
    if args.algo == 'bcfr_plus':
        from rlcard.agents.bcfr_plus_agent import BCFRPlusAgent
        agent = BCFRPlusAgent(
            env,
            args.m,
        )
    
    if args.algo == 'nfsp':
        agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=[64,64],
                q_mlp_layers=[64,64],
                device=device,
                save_path=args.log_dir,
            )
        from rlcard.agents import RandomAgent
        env.set_agents([agent, RandomAgent(num_actions=env.num_actions)])
    if args.algo == 'dqn':
        from rlcard.agents import DQNAgent
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
            save_path=args.log_dir,
        )
        from rlcard.agents import RandomAgent
        env.set_agents([agent, RandomAgent(num_actions=env.num_actions)])

    # 测试
    if args.adversary == 'random':
        from rlcard.agents import RandomAgent
        agent1 = RandomAgent(num_actions=env.num_actions)
    
    eval_env.set_agents([
        agent,
        agent1,
    ])
    # data = {}
    p = [1/args.m for _ in range(args.m)]
    pr = [0.3,0.3,0.4]
    indices = [0, 1, 2] 
    data = [0 for _ in range(env.state_shape[0][0])]
    all_data = [0 for _ in range(env.state_shape[0][0]+env.num_players)]
    # data_num = 0
    num = 1
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            if episode == 0:
                typ= args.typ
            else:
                typ = random.choices(indices, weights=pr)[0]
            if episode % args.evaluate_every == 0:
                trajectories, payoffs = eval_env.run(is_training=False, typ=typ)
                # print(payoffs)
                data = [0 for _ in range(env.state_shape[0][0]+env.num_players)]
                
                for i in range(len(trajectories)):
                    for j in range(len(trajectories[i])):
                        if type(trajectories[i][j])==dict:
                            num += 1
                            for q in range(len(trajectories[i][j]['obs'])):
                                data[q] += trajectories[i][j]['obs'][q]
                        # elif type(trajectories[i][j])==int:
                        #     data[env.state_shape[0][0]+trajectories[i][j]] += 1
                # for i in range(len(data)):
                    # data[i] /= num
                # data.append(num)
                for i in range(len(payoffs)):
                    data[env.state_shape[0][0]+i] = payoffs[i]
                all_data = [all_data[i]+data[i] for i in range(len(all_data))]
            
            
            if args.algo == 'nfsp':
                agent.sample_episode_policy()

            if args.algo == 'bcfr' or args.algo == 'deep_bcfr' or args.algo == 'bcfr_plus':
                tmp_data = []
                for i in range(len(all_data)):
                    tmp_data.append(all_data[i]/num)
                p = agent.train(p,tmp_data,episode)
                print(p)
                # print(agent.train(p,tmp_data),p)
            elif args.algo == 'cfr' or args.algo == 'deep_cfr' or args.algo == 'mccfr' or args.algo == 'cfr_plus':
                agent.train()
            elif args.algo == 'nfsp' or args.algo == 'dqn':
                typ = random.randint(0, args.m-1)
                trajectories, payoffs = env.run(is_training=False,typ=typ)
                trajectories = reorganize(trajectories, payoffs)
                for ts in trajectories[0]:
                    agent.feed(ts)
            print('\rIteration {}'.format(episode), end='')

            e = exploitability(env, agent, 50,typ=typ)
            reward = tournament(eval_env,args.num_eval_games,typ)[0]
            

            # if episode % args.evaluate_every == 0:
            #     agent.save()
            #     reward = tournament(eval_env, args.num_eval_games)[0]
            #     logger.log_performance(episode, reward)
            #     data[episode] = reward
            
            writer.add_scalar('exploitability', e, episode)
            writer.add_scalar('eval_reward', reward, episode)
        # plot_curve(data, args.log_dir)


if __name__ == '__main__':
    args = get_args()
    train(args)