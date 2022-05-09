import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from collections import OrderedDict
from workspace_utils import active_session
import numpy as np
from PIL import Image

"""
1. Train
Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu

"""

#Command line input into scripts
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = "./flowers/")
    parser.add_argument('--checkpoint_file', type = str, default = "./checkpoint.pth")
    parser.add_argument('--arch', type = str, default = "vgg19")
    parser.add_argument('--learning_rate',type = float, default = 0.001)
    parser.add_argument('--hidden_units', type = int, default = 4096)
    parser.add_argument('--epochs',type=int, default = 5)
    parser.add_argument('--dropout', type = float, default = 0.3)
    parser.add_argument('--gpu',type = str, default = "gpu")
    parser.add_argument('--num_in_features',type = int, default = 25088)
    parser.add_argument('--outputs',type = int, default = 102)
    
    return parser.parse_args()

#Build the network
def load_model(arch,num_in_features,hidden_units,dropout,outputs,learning_rate):
    if arch == "vgg19":
        model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    #number of outputs = number of classes    

    model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(num_in_features, hidden_units)),
                            ('drop', nn.Dropout(p=dropout)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(hidden_units, outputs)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    #define criterion
    criterion = nn.NLLLoss()

    #define optimizer and only train the classifier parameters
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = optim.Adam(params_to_update, lr = learning_rate)
    
    return model

#Train your network
def train_model(epochs,trainloader,model,validloader):
    criterion = nn.NLLLoss()
    
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            
    optimizer = optim.Adam(params_to_update, lr = 0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    if torch.cuda.is_available():
        model.cuda()
     
    
# do long-running work here
# One epoch is forward and backward pass of all training samples 
# Batch size is number of traning samples in each forward and back pass

    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 100


    with active_session():
        for epoch in range(epochs):
            for inputs,labels in trainloader:
                steps += 1
            
                #Moving inputs and label tensors to default device (GPU if GPU is available else CPU)       
                inputs,labels = inputs.to(device),labels.to(device)
                logps = model.forward(inputs)
                loss = criterion(logps,labels)
            
                #Need to clear gradients and parameters in the optimizers for each training passes to train well
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                #loss.item() contains loss of sub-batches
                running_loss += loss.item()
            
                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                
                    #model.eval() turns off drop off during validation and testing
                    model.eval()
             
                    with torch.no_grad():
                        for inputs,labels in validloader:
                            inputs, labels = inputs.to(device),labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps,labels)
                            valid_loss += batch_loss.item()
                    
                            #Calculate accuracy
                            ps = torch.exp(logps)
                            top_p,top_class = ps.topk(1,dim=1)
                            equality = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {valid_loss/len(validloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(validloader):.3f}")
            
                    running_loss = 0
                    model.train()

def save_checkpoint(num_in_features,outputs,checkpoint_file,train_data,model):
    

# Save the mapping of flower labels (1-102) to array indices (0-101)
    model.class_to_idx = train_data.class_to_idx

# When loading the checkpoint, model has to be exactly as it was when it was trained
    checkpoint = {'input size': num_in_features,
                  'output size': outputs,
                  'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier}
                  

    # Done: Save the checkpoint 
    torch.save(checkpoint, checkpoint_file)

"""
1. Define the transformation on the training, validation and testing sets
2. Load the datasets using Image Folder
3. Using the image datasets and the trainforms, define the dataloaders
4. Load the model
5. Define the loss
6. Define the optimizer
7. Train the model
8. Save the model
"""
    
def main():
    args = parse_args()
    data_dir = args.data_dir
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    
    # DONE: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])])
    
    # DONE: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform = train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform = valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform = test_transforms)


    # DONE: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    #Load model
    model = load_model(args.arch,args.num_in_features,args.hidden_units,args.dropout,args.outputs,args.learning_rate)
    
    #train model
    train_model(args.epochs,trainloader,model,validloader)
    
    #save checkpoint
    save_checkpoint(args.num_in_features, args.outputs,args.checkpoint_file,train_data,model)
    
if __name__ == "__main__":
    main()
    
    
    
    