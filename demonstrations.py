import os
import numpy as np
import gymnasium as gym
import time
import pygame
import re
from sdc_wrapper import SDC_Wrapper
import os, re, math
import numpy as np
from numpy.lib.format import open_memmap
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_demonstrations(data_folder, from_one_file=False):
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
    
    if not from_one_file:
        N = int((len([name for name in os.listdir(data_folder) if os.path.isfile(data_folder + "/" + name)]) - 3 )/ 2)
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
    
    else:
        observations = np.load(data_folder + "/observations.npy", allow_pickle=True)
        actions = np.load(data_folder + "/actions.npy", allow_pickle=True)

        observations = [observations[i] for i in range(observations.shape[0])]
        actions = [actions[i] for i in range(actions.shape[0])]

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
        m = int((len([name for name in os.listdir(data_folder) if os.path.isfile(data_folder + "/" + name)]) - 3 )/ 2)

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


def merge_demonstrations(
    data_folder,
    out_actions="actions_1.npy",
    out_observations="observations_1.npy",
    dtype=None,
):
    """
    Minimal merge for .npy files only:
      - Finds action_<id>.npy and observation<id>.npy
      - Keeps ids present in BOTH
      - Loads everything into memory, stacks on axis 0, saves two .npy files
    """
    action_rx = re.compile(r"action_(\d+)\.npy$")
    obs_rx    = re.compile(r"observation(\d+)\.npy$")

    action_files = {}
    observation_files = {}

    for name in os.listdir(data_folder):
        path = os.path.join(data_folder, name)
        if not os.path.isfile(path):
            continue
        m = action_rx.fullmatch(name)
        if m:
            action_files[int(m.group(1))] = path
            continue
        m = obs_rx.fullmatch(name)
        if m:
            observation_files[int(m.group(1))] = path

    ids = sorted(set(action_files).intersection(observation_files))
    if not ids:
        raise ValueError(f"No matching action/observation pairs found in {data_folder}.")

    # Load first pair to fix shapes/dtypes
    first_a = np.load(action_files[ids[0]], allow_pickle=False)
    first_o = np.load(observation_files[ids[0]], allow_pickle=False)

    if dtype is not None:
        a_dtype = np.dtype(dtype)
        o_dtype = np.dtype(dtype)
        first_a = first_a.astype(a_dtype, copy=False)
        first_o = first_o.astype(o_dtype, copy=False)
    else:
        a_dtype = first_a.dtype
        o_dtype = first_o.dtype

    a_shape = first_a.shape
    o_shape = first_o.shape

    actions = [first_a]
    observations = [first_o]

    # Load remaining pairs
    for i in ids[1:]:
        if i % 1000 == 0:
            print(f"Processing id {i}...")
        a = np.load(action_files[i], allow_pickle=False)
        o = np.load(observation_files[i], allow_pickle=False)

        if a.shape != a_shape or o.shape != o_shape:
            raise ValueError(
                f"Shape mismatch at id {i}. "
                f"Expected action {a_shape}, observation {o_shape} but got {a.shape}, {o.shape}."
            )

        if dtype is not None:
            a = a.astype(a_dtype, copy=False)
            o = o.astype(o_dtype, copy=False)

        actions.append(a)
        observations.append(o)

    actions_arr = np.stack(actions, axis=0)
    observations_arr = np.stack(observations, axis=0)

    out_a_path = os.path.join(data_folder, out_actions)
    out_o_path = os.path.join(data_folder, out_observations)
    np.save(out_a_path, actions_arr)
    np.save(out_o_path, observations_arr)

    return out_a_path, out_o_path


def merge_two_npy_files(file1, file2, output_file):
    """
    Merge two .npy files by loading them, concatenating their contents along
    the first axis, and saving the result to a new .npy file.
    """
    data1 = np.load(file1)
    data2 = np.load(file2)

    merged_data = np.concatenate((data1, data2), axis=0)
    np.save(output_file, merged_data)



if __name__ == "__main__":
    merge_demonstrations('/media/sn/Frieder_Data/Master_Machine_Learning/SDC/SDC/data_HQ')
    #merge_two_npy_files('/media/sn/Frieder_Data/Master_Machine_Learning/SDC/SDC/new_data_cut_101125/actions_frieder.npy', '/media/sn/Frieder_Data/Master_Machine_Learning/SDC/SDC/new_data_cut_101125/new_data3/actions_1.npy', '/media/sn/Frieder_Data/Master_Machine_Learning/SDC/SDC/new_data_cut_101125/actions_full.npy')
    #merge_two_npy_files('/media/sn/Frieder_Data/Master_Machine_Learning/SDC/SDC/new_data_cut_101125/observations_frieder.npy', '/media/sn/Frieder_Data/Master_Machine_Learning/SDC/SDC/new_data_cut_101125/new_data3/observations_1.npy', '/media/sn/Frieder_Data/Master_Machine_Learning/SDC/SDC/new_data_cut_101125/observations_full.npy')
    #merge_two_npy_files('./data/observations_1.npy', './data/observations_2.npy', './data/observations.npy')
    #actions = np.load('/media/sn/Frieder_Data/Master_Machine_Learning/SDC/SDC/new_data_cut_101125/actions_full.npy')
    #print(actions.shape)
    #observations, actions = load_demonstrations("/home/stud217/Ex1/template/data")
    #actions = np.stack(actions) 
    #steering_unique = np.unique(actions[:,0])
    #gas_unique = np.unique(actions[:,1])
    #break_unique = np.unique(actions[:,2])
    #print(steering_unique)
    #print(gas_unique)
    #print(break_unique)