import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import numpy as np
import torch
import gymnasium as gym
import argparse

from training import train
from demonstrations import record_demonstrations
from sdc_wrapper import SDC_Wrapper

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False


def evaluate(args, trained_network_file):
    """
    """
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_action = torch.load(trained_network_file, map_location=device)
    infer_action.eval()

    render_mode = 'rgb_array' if args.no_display else 'human'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode), remove_score=True,
                      return_linear_velocity=False)

    infer_action = infer_action.to(device)

    for episode in range(5):
        try:
            observation, _ = env.reset()
        except:
            print("Please note that you can't use the window on the cluster.")
            return

        reward_per_episode = 0
        for t in range(600):
            action_scores = infer_action(torch.Tensor(
                np.ascontiguousarray(observation[None])).to(device))

            steer, gas, brake = infer_action.scores_to_action(action_scores)

            observation, reward, done, trunc, info = env.step([steer, gas, brake])
            reward_per_episode += reward

        print('episode %d \t reward %f' % (episode, reward_per_episode))

    env.close()


def calculate_score_for_leaderboard(args, trained_network_file):
    """
    Evaluate the performance of the network. This is the function to be used for
    the final ranking on the course-wide leader-board, only with a different set
    of seeds. Better not change it.
    """
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    infer_action = torch.load(trained_network_file, map_location=device)
    infer_action.eval()

    render_mode = 'rgb_array' if args.no_display else 'human'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode), remove_score=True,
                      return_linear_velocity=False)

    seeds = [22597174, 68545857, 75568192, 91140053, 86018367,
             49636746, 66759182, 91294619, 84274995, 31531469]

    total_reward = 0

    with torch.no_grad():
        for episode, seed in enumerate(seeds):
            try:
                observation, _ = env.reset(seed=seed)
            except:
                print("Please note that you can't use the window on the cluster.")
                return

            reward_per_episode = 0
            for t in range(600):
                action_scores = infer_action(torch.Tensor(
                    np.ascontiguousarray(observation[None])).to(device))

                steer, gas, brake = infer_action.scores_to_action(action_scores)
                observation, reward, done, trunc, info = env.step([steer, gas, brake])
                reward_per_episode += reward

            print('episode %d \t reward %f' % (episode, reward_per_episode))
            total_reward += np.clip(reward_per_episode, 0, np.infty)

    print('---------------------------')
    print(' total score: %f' % (total_reward / len(seeds)))
    print('---------------------------')
    env.close()


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "--train",
        action="store_true",
    )
    main_parser.add_argument(
        "--test",
        action="store_true",
    )
    main_parser.add_argument(
        "--score",
        action="store_true",
    )
    main_parser.add_argument(
        "--collect",
        action="store_true",
    )
    main_parser.add_argument(
        "--agent_load_path",
        type=str,
        default="agent.pth",
        help="Path to the .pth file of the trained agent."
    )
    main_parser.add_argument(
        "--agent_save_path",
        type=str,
        default="agent.pth",
        help="Save path of the trained model."
    )
    main_parser.add_argument(
        "--training_data_path",
        type=str,
        default="data",
        help="Save path of the trained model."
    )

    main_parser.add_argument(
        "--nr_epochs",
        type=int,
        default=75,
        help="Number of training epochs."
    )
    main_parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    main_parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate."
    )
    main_parser.add_argument(
        "--no_display",
        action="store_true",
        default=False
    )

    args = main_parser.parse_args()

    if args.collect:
        print('Collect: You can collect training data now.')
        record_demonstrations(args.training_data_path)
    elif args.train:
        print('Train: Training your network with the collected data.')
        train(args.training_data_path, args.agent_save_path, args)
    elif args.test:
        print('Test: Your trained model will be tested now.')
        evaluate(args, args.agent_load_path)
    else:
        calculate_score_for_leaderboard(args, args.agent_load_path)

