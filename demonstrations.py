import os
import numpy as np
import gymnasium as gym
import time
import pygame
import re
from sdc_wrapper import SDC_Wrapper

def load_demonstrations(data_folder):
    """
    1.1 a)
    Given the folder containing the expert demonstrations, the data gets loaded and
    stored it in two lists: observations and actions.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    return:
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    
    N = int((len([name for name in os.listdir(data_folder) if os.path.isfile(data_folder + "/" + name)]) - 1 )/ 2)
    print(f"{N=}")
    observations = []
    actions = []

    for i in range(N):
        observation = np.load(data_folder + f"/observation{i}.npy")
        action      = np.load(data_folder + f"/action_{i}.npy")

        observations.append(observation)
        actions.append(action)

    np_actions = np.stack(actions)
    print(np.unique(np_actions[:, 0]))
    print(np.unique(np_actions[:, 1]))
    print(np.unique(np_actions[:, 2]))

    return observations, actions

    


def save_demonstrations(data_folder, actions, observations, from_one_file=False):
    """
    1.1 f)
    Save the lists actions and observations in numpy .npy files that can be read
    by the function load_demonstrations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """

    if not from_one_file:
        m = int((len([name for name in os.listdir(data_folder) if os.path.isfile(data_folder + "/" + name)]) - 1 )/ 2)

        for i, (action, observation) in enumerate(zip(actions, observations)):
            np.save(data_folder + f"/action_{i+m}.npy", action)
            np.save(data_folder + f"/observation{i+m}.npy", observation)
    else:
        old_actions = np.load(data_folder + "/actions.npy")
        old_observations = np.load(data_folder + "/observations.npy")

        new_actions = np.asarray(actions)
        new_observations = np.asarray(observations)

        all_actions = np.concatenate([old_actions, new_actions], axis=0)
        all_observations = np.concatenate([old_observations, new_observations], axis=0)

        np.save(data_folder + f"/actions.npy", all_actions)
        np.save(data_folder + f"/observations.npy", all_observations)




    




class ControlStatus:
    """
    Class to keep track of key presses while recording demonstrations.
    """
    def __init__(self):
        self.stop = False
        self.save = False
        self.quit = False

        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.quit = True

            if event.type == pygame.KEYDOWN:
                self.key_press(event)

        keys = pygame.key.get_pressed()
        self.accelerate = 0.5 if keys[pygame.K_UP] else 0
        self.brake = 0.8 if keys[pygame.K_DOWN] else 0
        self.steer = 1 if keys[pygame.K_RIGHT] else (-1 if keys[pygame.K_LEFT] else 0)

    def key_press(self, event):
        if event.key == pygame.K_ESCAPE:    self.quit = True
        if event.key == pygame.K_SPACE:     self.stop = True
        if event.key == pygame.K_TAB:       self.save = True


def record_demonstrations(demonstrations_folder):
    """
    Function to record own demonstrations by driving the car in the gym car-racing
    environment.
    demonstrations_folder:  python string, the path to where the recorded demonstrations
                        are to be saved

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    """

    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='human'), remove_score=True, return_linear_velocity=False)
    try:
        _, _ = env.reset(seed=int(np.random.randint(0, 1e6)))
    except:
        print("Please note that you can't collect data on the cluster.")
        return

    status = ControlStatus()
    total_reward = 0.0

    while not status.quit:
        observations = []
        actions = []
        # get an observation from the environment
        observation, _ = env.reset()

        while not status.stop and not status.save and not status.quit:
            status.update()

            # collect all observations and actions
            observations.append(observation.copy())
            actions.append(np.array([status.steer, status.accelerate,
                                    status.brake]))
            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, trunc, info = env.step([status.steer,
                                                           status.accelerate,
                                                           status.brake])

            total_reward += reward
            time.sleep(0.01)

        if status.save:
            save_demonstrations(demonstrations_folder, actions, observations, from_one_file=True)
            status.save = False

        status.stop = False

    env.close()



def merge_demonstrations(data_folder):
    """
    Merge individual action/observation files into two aggregate .npz archives.

    The function expects files named action_<id>.npz (or .npy) and
    observation<id>.npz (or .npy). All ids that exist for both an action and an
    observation are loaded, ordered by their numeric id, stacked, and saved as
    actions.npz and observations.npz inside data_folder.
    """
    action_pattern = re.compile(r"action_(\d+)\.(npy|npz)$")
    observation_pattern = re.compile(r"observation(\d+)\.(npy|npz)$")

    action_files = {}
    observation_files = {}

    for filename in os.listdir(data_folder):
        action_match = action_pattern.fullmatch(filename)
        if action_match:
            action_files[int(action_match.group(1))] = os.path.join(data_folder, filename)
            continue
        observation_match = observation_pattern.fullmatch(filename)
        if observation_match:
            observation_files[int(observation_match.group(1))] = os.path.join(data_folder, filename)

    paired_ids = sorted(set(action_files) & set(observation_files))
    if not paired_ids:
        raise ValueError(f"No matching action/observation file pairs found in {data_folder}.")

    actions = []
    observations = []
    for idx in paired_ids:
        actions.append(np.load(action_files[idx]))
        observations.append(np.load(observation_files[idx]))

    actions_array = np.stack(actions)
    observations_array = np.stack(observations)

    np.save(os.path.join(data_folder, 'actions.npy'), actions_array)
    np.save(os.path.join(data_folder, 'observations.npy'), observations_array)



if __name__ == "__main__":
    merge_demonstrations('./data')
    #observations, actions = load_demonstrations("/home/stud217/Ex1/template/data")
    #actions = np.stack(actions) 
    #steering_unique = np.unique(actions[:,0])
    #gas_unique = np.unique(actions[:,1])
    #break_unique = np.unique(actions[:,2])
    #print(steering_unique)
    #print(gas_unique)
    #print(break_unique)