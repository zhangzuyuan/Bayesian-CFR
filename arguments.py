import argparse
import os
import time

def get_args():
    parser = argparse.ArgumentParser("DeepCFR")
    # 训练环境
    parser.add_argument("--env", type=str, default="leduc-holdem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--num_eval_games", type=int, default=2000)
    parser.add_argument("--evaluate_every", type=int, default=2)
    parser.add_argument("--algo", type=str, default="cfr")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument('--run_name', default='run_name', help='Name to identify the experiments')

    parser.add_argument("--adversary", type=str, default="random")
    # Bayesian
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--typ", type=int, default=1)

    
    args = parser.parse_args()
    args.run_name = "{}_{}_{}_{}".format(args.env,args.algo, args.m, time.strftime("%Y%m%dT%H%M%S"))
    args.log_dir = os.path.join(
        f'{args.log_dir}',
        args.run_name
    )
    return args