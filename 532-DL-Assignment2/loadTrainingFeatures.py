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
print("hs loaded modules\n")


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
print("hs loaded datasets\n")



class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.modified_vgg_model = models.vgg16(pretrained=True).cuda()
        self.modified_vgg_model.classifier = nn.Sequential(
            *(self.modified_vgg_model.classifier[i] for i in range(5)))

    def forward(self, images):
        return self.modified_vgg_model(images)



modified_vgg_model = MyModel()
modified_vgg_model.eval()

print("hs loaded the model\n")



# First we vectorize all of the features of training images and write the results to a file.

# -- Your code goes here --
training_vectors = []
for ind, image_id in enumerate(train_ids[:]):
    img = load_image(train_id_to_file[image_id])
    
    '''
    print("\nTrIm: ", image_id)
    image = Image.open(train_id_to_file[image_id]).convert('RGB')
    plt.imshow(image)
    plt.figure(image_id)
    plt.show()
    '''
    
    out_features = modified_vgg_model(img).squeeze()
    
    temp = out_features.cpu().data
    temp = temp.numpy()
    training_vectors.append(temp)

    print("hs loaded image: ", ind,"\n")

print("Done saving ", len(training_vectors), "  images\n")
np.save(open('outputs/training_vectors', 'wb+'), training_vectors)