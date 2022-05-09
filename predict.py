import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
from collections import OrderedDict
from PIL import Image
import argparse
import json

"""
2. Predict
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type = str, default = './checkpoint.pth')
    parser.add_argument('--test_image', type = str, default = 'flowers/test/1/image_06743.jpg')
    parser.add_argument('--top_k', type = int, default = 5)
    parser.add_argument('--cat_name', type = str, default = 'cat_to_name.json')
    
    return parser.parse_args()

#1 load in a mapping from category label to category name
def load_cat_to_name(cat_name):
    with open(cat_name, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def loading_model(checkpoint_file):
    
    checkpoint = torch.load(checkpoint_file)
    
    model = models.vgg19(pretrained=True)
    model.name = "vgg19"
    
    for param in model.parameters(): 
        param.requires_grad = False

    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    #model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier']


    return model

def process_image(image_path):
    # Load Image
    img = Image.open(image_path)
    img = img.resize((256,256))
    value = 0.5*(256-224)
    img = img.crop((value,value,256-value,256-value))

    
    # convert image into numpy array
    img = np.array(img)
    
    #normalized image:  [0.485, 0.456, 0.406] for mean and [0.229, 0.224, 0.225] for the standard deviations
    np_img = np.array(img) /225
    np_img = (np_img - np.array([0.485,0.456,0.406])/np.array([0.229,0.224,0.225]))
    np_img = np_img.transpose(2,0,1)
    

    return np_img


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    #Load the image and use the function that preprocessess the image
    image = process_image(image_path)
        
    #Returns a new tensor with a dimension of size one inserted at the specified position.
    image = torch.from_numpy(np.array([image])).float()
    #image = Variable(image)
    image = image.to(device)

    with torch.no_grad():
        output = model.forward(image)

    output_p = torch.exp(output).data
    probs, classes = output_p.topk(topk)
    probs = probs.cpu().detach().numpy().tolist()[0] 
    classes = classes.cpu().detach().numpy().tolist()[0]

    #reverse key,value of dictionary and map to class using indices
    # previous mapping: model.class_to_idx = trainloader.class_to_idx
    
    idx_to_class = {value:key for key,value in model.class_to_idx.items()}    
    classes = [idx_to_class[i] for i in classes]
    
    return probs,classes

'''
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
'''

def main():
    args = parse_args()
    checkpoint_file = args.checkpoint_file
    test_image = args.test_image
    
    #Load the model
    model = loading_model(checkpoint_file)
    
    #Categorise Name
    cat_to_name = load_cat_to_name(args.cat_name)
 
    #predict model
    probs , classes = predict(args.test_image, model, topk=args.top_k)

    #Sanity Checking

    image = Image.open(test_image)

    # Get top probabilities and name of classes
    name_of_classes = [cat_to_name[i] for i in classes]
    print(name_of_classes)

    classes_probs_dicts = dict(zip(name_of_classes, probs))
    for i,j in classes_probs_dicts.items():
        print ("Probality  of test image being a {} is  {:.2f} %".format(i,j*100))
    
    
if __name__ == "__main__":
    main()
    
              
    