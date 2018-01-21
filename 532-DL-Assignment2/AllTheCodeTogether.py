from collections import defaultdict
from IPython import display
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms

import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch


########## START: Load and Transform Images ##########
# Define a global transformer to appropriately scale images and subsequently convert them to a Tensor.
img_size = 224
loader = transforms.Compose([
  transforms.Scale(img_size),
  transforms.CenterCrop(img_size),
  transforms.ToTensor(),
]) 
def load_image(filename):
    """
    Simple function to load and preprocess the image.

    1. Open the image.
    2. Scale/crop it and convert it to a float tensor.
    3. Convert it to a variable (all inputs to PyTorch models must be variables).
    4. Add another dimension to the start of the Tensor (b/c VGG expects a batch).
    5. Move the variable onto the GPU.
    """
    image = Image.open(filename).convert('RGB')
    image_tensor = loader(image).float()
    image_var = Variable(image_tensor).unsqueeze(0)
    return image_var.cuda()

load_image('data/train2014/COCO_train2014_000000000009.jpg')
############ END: Load and Transform Images ##########



########## START: Load COCO DATA AND ImageNet labels ##########
# Load ImageNet label to category name mapping.
imagenet_categories = list(json.load(open('data/imagenet_categories.json')).values())

# Load annotations file for the 100K training images.
mscoco_train = json.load(open('data/annotations/train2014.json'))
train_ids = [entry['id'] for entry in mscoco_train['images']]
train_id_to_file = {entry['id']: 'data/train2014/' + entry['file_name'] for entry in mscoco_train['images']}
category_to_name = {entry['id']: entry['name'] for entry in mscoco_train['categories']}
category_idx_to_name = [entry['name'] for entry in mscoco_train['categories']]
category_to_idx = {entry['id']: i for i,entry in enumerate(mscoco_train['categories'])}

# Load annotations file for the 100 validation images.
mscoco_val = json.load(open('data/annotations/val2014.json'))
val_ids = [entry['id'] for entry in mscoco_val['images']]
val_id_to_file = {entry['id']: 'data/val2014/' + entry['file_name'] for entry in mscoco_val['images']}

# We extract out all of the category labels for the images in the training set. We use a set to ignore 
# duplicate labels.
train_id_to_categories = defaultdict(set)
for entry in mscoco_train['annotations']:
    train_id_to_categories[entry['image_id']].add(entry['category_id'])

# We extract out all of the category labels for the images in the validation set. We use a set to ignore 
# duplicate labels.
val_id_to_categories = defaultdict(set)
for entry in mscoco_val['annotations']:
    val_id_to_categories[entry['image_id']].add(entry['category_id'])
############ END: Load COCO DATA AND ImageNet labels ##########



#########################################################################
###################### START: Load and use VGG16 ########################
#########################################################################
vgg_model = models.vgg16(pretrained=True).cuda()
vgg_model.eval()

# Let's see what the model looks like.
vgg_model


########## START: Making Predictions with VGG16 ########
softmax = nn.Softmax()
for image_id in val_ids[:10]:
    # Display the image.
    # -- Your code goes here --
    image = Image.open(val_id_to_file[image_id]).convert('RGB')
    plt.imshow(image)
    plt.figure(image_id)
    plt.show()
    

    # Print all of the category labels for this image.
    # -- Your code goes here --
    print("Actual Labels:")
    for i,category in enumerate(val_id_to_categories[image_id]):
        print("%d. %s" % (i, category_to_name[category]))
        
    print("\n")
  
    # Load/preprocess the image.
    img = load_image(val_id_to_file[image_id])

    # Run the image through the model and softmax.
    label_likelihoods = softmax(vgg_model(img)).squeeze()

    # Get the top 5 labels, and their corresponding likelihoods.
    probs, indices = label_likelihoods.topk(5)
        
    
    # Iterate and print out the predictions.
    # -- Your code goes here --
    print("Predicted Labels:")
    for p, i in zip(probs,indices):
        temp = i.cpu().data
        temp = temp.numpy()
        tempP = p.cpu().data
        tempP = tempP.numpy()
        print(imagenet_categories[int(temp)], "   **prob: ", float(tempP))
############ END: Making Predictions with VGG16 ########