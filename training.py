import torch
import torch.nn as nn
import torchvision
import random
import time
from network import ClassificationNetwork
from demonstrations import load_demonstrations
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

def train(data_folder, trained_network_file, args):
    """
    Function for training the network.
    """
    infer_action = ClassificationNetwork()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    #Loss function
    loss_function = nn.CrossEntropyLoss()

    observations, actions = load_demonstrations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_classes(actions))]

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_action.to(device)

    nr_epochs = args.nr_epochs
    batch_size = args.batch_size
    start_time = time.time()

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
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
                
                total_loss += loss

                batch_in = []
                batch_gt = []

        lr_scheduler.step()

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

    torch.save(infer_action, trained_network_file)