import torch
import torchvision
import numpy as np

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
                print(f"Unknown action: {action}. Apppend to class 'nothing'")
                classes.append(torch.Tensor([8]).type(torch.LongTensor))
                class_occurences[8] += 1

        assert len(classes) == len(actions) # sanity check

        uniform_class_weight = len(actions) / 9
        class_weights = [uniform_class_weight / occ if occ > 0 else 0 for occ in class_occurences]
        
        
        class_proportions = [prop / len(actions) for prop in class_occurences]
        class_names = ["left", "left+gas", "right", "right+gas", "left+brake", "right+brake", "gas", "brake", "nothing"]
        class_prop_dict = {class_names[i]: prop for i, prop in enumerate(class_proportions)}

        print("Class proportions:")
        print(class_prop_dict)

        return classes, class_weights


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
        
    
        
        if class_number == 0:
            return -1., 0., 0.
        
        elif class_number == 1:
            return -1., 0.5, 0.
        
        elif class_number == 2:
            return 1., 0., 0.
        
        elif class_number == 3:
            return 1., 0.5, 0.
        
        elif class_number == 4:
            return -1., 0., 0.8
        
        elif class_number == 5:
            return 1., 0., 0.8
        
        elif class_number == 6:
            return 0., 0.5, 0.
        
        elif class_number == 7:
            return 0., 0., 0.8
        
        elif class_number == 8:
            return 0., 0., 0.
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