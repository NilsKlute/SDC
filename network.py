import torch
import torchvision
import numpy as np
from collections import OrderedDict
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

class ClassificationNetwork(torch.nn.Module):
    def __init__(self, dropout=0.2):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()

        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = dropout
                

        #96x96x3 -> 96x96x8
        self.convLayer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.LeakyReLU(negative_slope =0.2),
            torch.nn.Dropout(p=self.dropout))
        
        #48x48x8 -> 48x48x16
        self.convLayer2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.LeakyReLU(negative_slope =0.2),
            torch.nn.Dropout(p=self.dropout))
        
        #24x24x16 -> 24x24x32
        self.convLayer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.LeakyReLU(negative_slope =0.2),
            torch.nn.Dropout(p=self.dropout))
        
        self.max_pool = torch.nn.MaxPool2d(2)

        self.lin_layer1 = torch.nn.Sequential(
            torch.nn.Linear(12*12*32,  1024),
            torch.nn.LeakyReLU( negative_slope =0.2))
        
        self.lin_layer2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 9),
            torch.nn.LeakyReLU( negative_slope =0.2))

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, C)
        """
        
        x = torch.torch.permute(observation, (0, 3, 1, 2))

        x = self.convLayer1(x)
        x = self.max_pool(x)

        x = self.convLayer2(x)
        x = self.max_pool(x)

        x = self.convLayer3(x)
        x = self.max_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.lin_layer1(x)
        x = self.lin_layer2(x)

        return x

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 1-dim vector
        with the entry corresponding to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 1
        """
        classes = []
        class_occurences = [0 for _ in range(9)]
        for action in actions:
            steer = round(action[0].item(), 2)
            gas = round(action[1].item(), 2)
            brake = round(action[2].item(), 2)

            # steer left
            if steer == -1 and gas == 0 and brake == 0:
                classes.append(torch.Tensor([0]).type(torch.LongTensor))
                class_occurences[0] += 1
            
            # steer left and gas
            elif steer == -1 and gas == 0.5 and brake == 0:
                classes.append(torch.Tensor([1]).type(torch.LongTensor))
                class_occurences[1] += 1

            # steer right
            elif steer == 1 and gas == 0 and brake == 0:
                classes.append(torch.Tensor([2]).type(torch.LongTensor))
                class_occurences[2] += 1
            
            # steer right and gas
            elif steer == 1 and gas == 0.5 and brake == 0:
                classes.append(torch.Tensor([3]).type(torch.LongTensor))
                class_occurences[3] += 1

            # steer left and brake
            elif steer == -1 and gas == 0 and brake == 0.8:
                classes.append(torch.Tensor([4]).type(torch.LongTensor))
                class_occurences[4] += 1

            # steer right and brake
            elif steer == 1 and gas == 0 and brake == 0.8:
                classes.append(torch.Tensor([5]).type(torch.LongTensor))
                class_occurences[5] += 1

            # gas
            elif steer == 0 and gas == 0.5 and brake == 0:
                classes.append(torch.Tensor([6]).type(torch.LongTensor))
                class_occurences[6] += 1

            # brake
            elif steer == 0 and gas == 0 and brake == 0.8:
                classes.append(torch.Tensor([7]).type(torch.LongTensor))
                class_occurences[7] += 1
            
            # nothing
            elif steer == 0 and gas == 0 and brake == 0:
                classes.append(torch.Tensor([8]).type(torch.LongTensor))
                class_occurences[8] += 1
            else:
                #print(f"Unknown action: {action}. Apppend to class 'nothing'")
                classes.append(torch.Tensor([8]).type(torch.LongTensor))
                class_occurences[8] += 1

        assert len(classes) == len(actions) # sanity check

        uniform_class_weight = len(actions) / 9
        #class_weights = [uniform_class_weight / occ if occ > 0 else 0 for occ in class_occurences]
        
        
        #class_proportions = [prop / len(actions) for prop in class_occurences]
        #class_names = ["left", "left+gas", "right", "right+gas", "left+brake", "right+brake", "gas", "brake", "nothing"]
        #class_prop_dict = {class_names[i]: prop for i, prop in enumerate(class_proportions)}

        #print("Class proportions:")
        #print(class_prop_dict)

        return classes#, class_weights


    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        prediction = scores[0]
        class_number = prediction.max(dim=0, keepdim=False)[1].item()
        
    
        speed = 0.35
        if class_number == 0:
            return -1., 0., 0.
        
        elif class_number == 1:
            return -1., speed, 0.
        
        elif class_number == 2:
            return 1., 0., 0.
        
        elif class_number == 3:
            return 1., speed, 0.
        
        elif class_number == 4:
            return -1., 0., 0.8
        
        elif class_number == 5:
            return 1., 0., 0.8
        
        elif class_number == 6:
            return 0., speed, 0.
        
        elif class_number == 7:
            return 0., 0., 0.8
        
        elif class_number == 8:
            return 0., speed, 0.
        else:
            print(f"Unknown Class number {class_number}")
            return 0., 0., 0.
        
    
        

    def extract_sensor_values(self, observation, batch_size):
        # just approximately normalized, usually this suffices.
        # can be changed by you
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255 / 5

        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1) / 255 / 10
        steer_crop[:, :10] *= -1
        steering = steer_crop.sum(dim=1, keepdim=True)

        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1) / 255 / 5
        gyro_crop[:, :14] *= -1
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)

        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
    

def state_dict_to_full_model(state_dict_path: str,
                             out_model_path: str,
                             dropout: float = 0.2,
                             map_location: str = "cpu",
                             strict: bool = True):
    """
    Load a saved state_dict/checkpoint, rebuild ClassificationNetwork, and save a full model.

    Args:
        state_dict_path: path to .pth file produced by torch.save(model.state_dict(), ...)
                         or a checkpoint dict that contains 'state_dict'.
        out_model_path:  where to write the full model (e.g., 'agent_full.pth').
        dropout:         dropout to use when recreating the model (must match training).
        map_location:    'cpu' (default) or e.g. 'cuda:0' to load tensors to a device.
        strict:          passed to load_state_dict; set False to ignore key mismatches.

    Returns:
        The instantiated model (already loaded with weights).
    """
    # 1) Load the file (could be OrderedDict or a dict with 'state_dict')
    ckpt = torch.load(state_dict_path, map_location=map_location)
    if isinstance(ckpt, OrderedDict):
        state_dict = ckpt
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

    # 2) Strip 'module.' prefix if it exists (DataParallel)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # 3) Recreate the SAME architecture as training
    model = ClassificationNetwork(dropout=dropout)

    # 4) Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        if missing:
            print(f"[load warning] Missing keys: {missing}")
        if unexpected:
            print(f"[load warning] Unexpected keys: {unexpected}")

    # 5) Save the full nn.Module so your eval code can torch.load(...).eval()
    torch.save(model, out_model_path)

    return model

if __name__ == "__main__":
    state_dict_to_full_model(
        state_dict_path="agent_dict.pth",
        out_model_path="agent.pth",)