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


###############################################################
########## START: Load COCO DATA AND ImageNet labels ##########
###############################################################
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


print("hs loaded COCO data\n");




###############################################################
############ START: Prepare the training data set #############
###############################################################
#my data class to load the dataset
class hsDatasetCl(torch.utils.data.Dataset):

    def __init__(self, dataIds):
        self.dataIds = dataIds
        self.transform = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
    def load_imageHS(self, filename):
        image = Image.open(filename).convert('RGB')
        image_tensor = self.transform(image).float()
        #image_var = Variable(image_tensor).unsqueeze(0)
        return image_tensor

    def __len__(self):
        return len(self.dataIds)

    def __getitem__(self, idx):
        return self.load_imageHS(train_id_to_file[self.dataIds[idx]])
    
    
#hs unit test
'''
trainingset = hsDatasetCl(train_ids)
print(len(trainingset))
print(trainingset[4])
'''

#load training data
trainingset = hsDatasetCl(train_ids)
#create labels
yy = []
for i in range(82783):
    traImageId = train_ids[i]
    cats = train_id_to_categories[traImageId]
    
    label = np.zeros(91, dtype=int)
    for val in cats:
        label[val] = 1
        
    yy.append(label)
    
y = np.array(yy)


#define a dataLoader for data and labels
trainingsetLoader = torch.utils.data.DataLoader(trainingset, batch_size=50,
                        shuffle=False, num_workers=2)

trainingsetLabelLoader = torch.utils.data.DataLoader(y, batch_size=50,
                        shuffle=False, num_workers=2)


#hs unit test
'''
print("trainingsetData size: ", len(trainingset))
print("first image size: ")
trainingset[0].size()
print("trainingsetLabels size: ", len(y))
for i, data in enumerate(trainingsetLoader, 0):
        # get the inputs
        inputs = data
     #   print(inputs.size())
        break

print("first image labels: ")
print(y[0])

print("first batch labels: ")
for i, data in enumerate(trainingsetLabelLoader, 0):
        # get the inputs
        inputs = data
       # print(inputs[0])
        break
'''
############## END: Prepare the training data set #############


print("hs loaded training set data\n");


###############################################################
############## START: Define and load the model ###############
###############################################################
#redefine the model NOT PUT ON GPU
# First we construct a class for the model
# -- Your code goes here --
class MyEndToEndModel(nn.Module):

    def __init__(self):
        super(MyEndToEndModel, self).__init__()
        self.endToEnd_model = models.vgg16(pretrained=True)
        self.endToEnd_model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Linear(512, 91),
        )
        
    def forward(self, images):
        x = self.endToEnd_model.features(images)
        x = x.view(x.size(0), -1)
        x = self.endToEnd_model.classifier(x)
        return x
    
    def save_checkpoint(self, state, is_best, filename='MyEndToEndModel_checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'MyEndToEndModel_best.pth.tar')


model = MyEndToEndModel()
model.eval()
model.cuda() #remove .cuda to not put model on GPU can use .cuda(device_id=0) for specific gpu
################ END: Define and load the model ###############


print("hs loaded model\n");


###############################################################
################# START: Training method ######################
###############################################################
criterion = torch.nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

softmax = nn.Softmax()
for epoch in range(2):  # loop over the dataset multiple times
    for i, (data, label) in enumerate(zip(trainingsetLoader, trainingsetLabelLoader)):
        # wrap the data and labels from each batch in Variables
        inputs = Variable(data.cuda()) #remove .cuda to not put model on GPU
        labels = Variable(label.cuda()).float() #remove .cuda to not put model on GPU

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = softmax(model(inputs))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        print("epoch: ",epoch, ", batch: ", i,", produced loss of", loss.data[0])

        model.save_checkpoint({
            'epoch': epoch,
            'batch': i,
            'state_dict': model.state_dict(),
            'val_error': 43,
            'optimizer' : optimizer.state_dict(),
        }, is_best=False)

        print('Saved learned parameters')

print('Finished Training')
################### END: Training method ######################


'''
###############################################################
############## START: Loading Saved Models ####################
###############################################################

import os
if os.path.isfile("checkpoint.pth.tar"):
            print("=> loading checkpoint '{}'".format("checkpoint.pth.tar"))
            checkpoint = torch.load("checkpoint.pth.tar")
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) (batch {}) (error {})"
                  .format("checkpoint.pth.tar", checkpoint['epoch'], checkpoint['batch'], checkpoint['val_error']))

################ END: Loading Saved Models ####################



###############################################################
################# START: Printing losses ######################
###############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


losses = {'train':[], 'validation':[]}

for i in range(40):
    losses['train'].append(i)
    losses['validation'].append(i*2)


plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
################### END: Printing losses ######################
'''