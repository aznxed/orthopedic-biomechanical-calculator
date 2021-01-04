import argparse
import json
import os
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# imports the model in model.py by name
from model import MulticlassClassification

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MulticlassClassification(model_info['input_features'], model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def get_class_distribution(obj):
    count_dict = {
        "Normal": 0,
        "Hernia": 0,
        "Spondylolisthesis": 0,
    }
    
    for i in obj:
        if i == 0: 
            count_dict['Normal'] += 1
        elif i == 1: 
            count_dict['Hernia'] += 1
        elif i == 2: 
            count_dict['Spondylolisthesis'] += 1          
        else:
            print("Check classes.")
            
    return count_dict

# Gets training data in batches from the train.csv file
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()
    print(train_y.shape)
    
    train_dataset = ClassifierDataset(train_x, train_y)
    
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list, dtype=torch.long)
    target_list = target_list[torch.randperm(len(target_list))]
    
    class_count = [i for i in get_class_distribution(train_y).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    print(class_weights)
    
    class_weights_all = class_weights[target_list]
    
    weighted_sampler = WeightedRandomSampler(weights=class_weights_all,
                                             num_samples=len(class_weights_all),
                                             replacement=True)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=12,
                              drop_last=True,
                              sampler=weighted_sampler)
    return train_loader, class_weights


# Provided training function
def train(model, train_loader, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """

    # training loop is provided
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        total_loss = 0

        for batch in train_loader:
            # get data
            batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device, dtype=torch.int64)

            optimizer.zero_grad()

            # get predictions from model
            y_pred = model(batch_x)

            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
        
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))


## TODO: Complete the main code
if __name__ == '__main__':

    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job

    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=int, default=.005, metavar='LR',
                        help='learning rate (default: .005)')

    # Model Parameters
    parser.add_argument('--input_features', type=int, default=10, metavar='IN', 
                        help='number of initial input features (default:  10)')
    
    parser.add_argument('--output_dim', type=int, default=1, metavar='O',
                        help='dimension of model output (default: 1)')
    

    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader, class_weights = _get_train_data_loader(args.batch_size, args.data_dir)

    print(type(train_loader))
    print(type(class_weights))
    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = MulticlassClassification(args.input_features, args.output_dim).to(device)

    #Define an optimizer and loss function for training
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, args.epochs, criterion, optimizer, device)


    # Keep the keys of this dictionary as they are 
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'output_dim': args.output_dim,
        }
        torch.save(model_info, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)