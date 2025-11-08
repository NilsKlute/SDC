import torch
import torch.nn as nn
import torchvision
import random
import time
import matplotlib.pyplot as plt
import os
from network import ClassificationNetwork
from demonstrations import load_demonstrations
import numpy as np
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

# SPECIFY HYPERPARAMETERS HERE   
nr_conv_layers = 3
nr_linear_layers = 2
data_augmentation = False
use_dropout = True
# These cofigurations will be used for grid search
dropout_params = [0.2]
lr_params = [1e-3]
batchsize_params = [256]
gamma_params = [0.5]

if not use_dropout:
    dropout_params = [0.0]

dataset_prop_params = [1] # specify proportions to use for training here (e.g., [0.1, 0.5, 1] for 10%, 50%, and 100% of the dataset)

def train(data_folder, trained_network_file, args):
    """
    Function for training the network.
    """
    # initialize auxiliary network for processing images and inferring actions
    infer_action = ClassificationNetwork()

    # setting device on GPU if available, else CPU 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # load dataset
    observations, actions = load_demonstrations(data_folder, from_one_file=True)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    # Data augementation
    if data_augmentation:
        augmented_observations = []
        augmented_actions = []

        transform_hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        transform_brightness = torchvision.transforms.ColorJitter(brightness=0.5)

        # Increase the datasetsize by a factor of 3
        for obs, act in zip(observations, actions):
            # assume obs is a float tensor in [0, 255] shaped (H, W, C)
            # 1) keep original
            augmented_observations.append(obs)
            augmented_actions.append(act)

            # 2) horizontal flip
            obs_chw = obs.permute(2, 0, 1)               # (C, H, W)
            hflipped_chw = transform_hflip(obs_chw)
            hflipped_obs = hflipped_chw.permute(1, 2, 0)  # back to (H, W, C)

            hflipped_act = act.clone()
            hflipped_act[0] = -hflipped_act[0]
            augmented_observations.append(hflipped_obs)
            augmented_actions.append(hflipped_act)

            # 3) brightness jitter
            # normalize to [0,1], apply jitter, then scale back
            obs_chw_01 = obs_chw / 255.0
            bright_chw_01 = transform_brightness(obs_chw_01)
            bright_chw = (bright_chw_01 * 255.0).clamp(0, 255)
            bright_obs = bright_chw.permute(1, 2, 0)

            augmented_observations.append(bright_obs)
            augmented_actions.append(act)

        observations = augmented_observations
        actions = augmented_actions
        print(f"Data augmentation applied. New dataset size: {len(observations)}")


    # get action classes and class weights
    action_classes, class_weights = infer_action.actions_to_classes(actions)
    class_weights_tensor = torch.tensor(class_weights).to(device)

    # define loss function with class weights
    loss_function = nn.CrossEntropyLoss(weight=class_weights_tensor)


    # create databatches
    all_batches = [batch for batch in zip(observations, action_classes)]
    random.shuffle(all_batches)
    dataset_size = len(all_batches)

    for dropout in dropout_params:
        for lr in lr_params:
            for batch_size in batchsize_params:
                for dataset_prop in dataset_prop_params:
                    for gamma in gamma_params:
                        
                        #----- create model folder using hyperparameter values -----#
                        model_folder = f"models/hyperconfig_dataset:{dataset_size}_bs:{batch_size}_conv_n:{nr_conv_layers}_lin_n:{nr_linear_layers}_aug={data_augmentation}_drop={dropout}_epochs:{args.nr_epochs}_lr:{lr}_gamma:{gamma}"
                        if not os.path.exists(model_folder):
                            os.makedirs(model_folder)
                        trained_network_file = os.path.join(model_folder, trained_network_file)


                        #-----  initialize network for training -----#
                        infer_action = ClassificationNetwork(dropout)
                        optimizer = torch.optim.Adam(infer_action.parameters(), lr=lr)
                        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=gamma)


                        #------- split training and validation data -------#
                        dataset_subset_size = int(dataset_prop * dataset_size)
                        batches = all_batches[:dataset_subset_size]
                        train_split_n = int(0.9 * len(batches))
                        train_batches = batches[:train_split_n]
                        val_batches = batches[train_split_n:]
                        
                        print(f"Training with dropout={dropout}, lr={lr}, batch_size={batch_size}, dataset_size={dataset_subset_size}, gamma={gamma}")
                        print("Number of training samples: ", len(train_batches))
                        print("Number of validation samples: ", len(val_batches))


                        #----- move network to GPU if available -----#
                        infer_action.to(device)


                        

                        #----- initialize early stopping parameters / loss saving ------#
                        best_epoch = 0
                        early_stopping_counter = 0
                        min_val_loss = float('inf')

                        train_val_loss_per_epoch = [] # to store train/val loss for each epoch
                        nr_epochs = args.nr_epochs
                        start_time = time.time()


                        # ------------------ training/validation loop ------------------ #

                        for epoch in range(nr_epochs):
                            if early_stopping_counter < 20:


                                #train epoch
                                random.shuffle(train_batches)
                                train_loss = 0
                                batch_in = []
                                batch_gt = []
                                optimizer_stepped = False
                                for batch_idx, batch in enumerate(train_batches):
                                    batch_in.append(batch[0].to(device))
                                    batch_gt.append(batch[1].to(device))

                                    if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                                        batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                                                (-1, 96, 96, 3))

                                        batch_gt = torch.reshape(torch.cat(batch_gt, dim=0), (-1, )).to(device)
                                        batch_out = infer_action(batch_in)
                                        loss = loss_function(batch_out, batch_gt)

                                        optimizer.zero_grad()
                                        loss.backward()
                                        optimizer.step()
                                        optimizer_stepped = True
                                        
                                        train_loss += loss.detach().cpu().item()

                                        batch_in = []
                                        batch_gt = []
                                
                                if optimizer_stepped:
                                    lr_scheduler.step()
                                


                                #validate epoch
                                with torch.no_grad():
                                    infer_action.eval()
                                    val_loss = 0

                                    batch_in = []
                                    batch_gt = []
                                    for batch_idx, batch in enumerate(val_batches):
                                        batch_in.append(batch[0].to(device))
                                        batch_gt.append(batch[1].to(device))

                                        if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                                            batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                                                    (-1, 96, 96, 3))

                                            batch_gt = torch.reshape(torch.cat(batch_gt, dim=0), (-1, )).to(device)
                                            torch.cuda.empty_cache()  # clear CUDA cache
                                            batch_out = infer_action(batch_in)
                                            loss = loss_function(batch_out, batch_gt)

                                            val_loss += loss.detach().cpu().item()

                                            batch_in = []
                                            batch_gt = []

                                    
                                    if val_loss < min_val_loss:
                                        min_val_loss = val_loss
                                        early_stopping_counter = 0
                                        best_epoch = epoch 
                                        torch.save(infer_action, trained_network_file)
                                    else:
                                        early_stopping_counter += 1

                                
                                # ----- record train/val loss and print progress ----- #
                                train_val_loss_per_epoch.append([train_loss / 9, val_loss])  #as train/val split is 90/10
                                infer_action.train()

                                time_per_epoch = (time.time() - start_time) / (epoch + 1)
                                time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
                                print("Epoch %5d\t[Train]\tloss: %.6f \t[Val]\tloss: %.6f \tETA: +%fs" % (
                                    epoch + 1, float(train_loss), float(val_loss), time_left))

                        print(f"Best epoch: {best_epoch + 1} with validation loss: {min_val_loss}")

                                
                        #plotting the training and validation loss and saving the figure
                        train_losses = [x[0] for x in train_val_loss_per_epoch]
                        val_losses = [x[1] for x in train_val_loss_per_epoch]
                        plt.plot(range(1, len(train_losses) +1), train_losses, label='Training Loss')
                        plt.plot(range(1, len(val_losses) +1), val_losses, label='Validation Loss')
                        plt.xlabel('Epochs')
                        plt.ylabel('Loss')
                        plt.title('Training and Validation Loss over Epochs')
                        plt.legend()
                        plt.savefig(os.path.join(model_folder, 'loss_plot.png'))

                        np.save(os.path.join(model_folder, 'train_val_loss.npy'), np.array(train_val_loss_per_epoch))
