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
import os
import shutil



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
############## START: Load Image Helper Method ################
###############################################################
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
    return image_var.cuda(device_id=0)
################ END: Load Image Helper Method ################





###############################################################
############## START: Define and load the model ###############
###############################################################
class MyEndToEndModel(nn.Module):

    def __init__(self):
        super(MyEndToEndModel, self).__init__()
        self.endToEnd_model = models.vgg16(pretrained=True)
        self.endToEnd_model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, 512),
            nn.Sigmoid(),
            nn.Linear(512, 91), 
            #I'm adding the last layer's activation manually to play around with different activations
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
# model.eval()  # use this when loading model just to predict
model.train()   # use this when loading model to train
model.cuda() #remove .cuda to not put model on GPU can use .cuda(device_id=0) for specific gpu
################ END: Define and load the model ###############


print("hs loaded model\n");


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
############## END: Prepare the training data set #############



###############################################################
############ START: Prepare the training data set #############
###############################################################
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
############## END: Prepare the training data set #############


print("hs loaded training set data\n");




###############################################################
########### START: Prepare the Validation data set ############
###############################################################
#create labels
yy_val = []
for i in range(100):
    valImageId = val_ids[i]
    cats = val_id_to_categories[valImageId]
    
    label = np.zeros(91, dtype=int)
    for val in cats:
        label[val] = 1
        
    yy_val.append(label)
    
y_val = np.array(yy_val)
############# END: Prepare the Validation data set ############


print("hs loaded validation set data\n");



###############################################################
################# START: Training method ######################
###############################################################
def train(model, learning_rate=0.001, batch_size=50, epochs=2):
    #define a dataLoader for training data and labels
    trainingsetLoader = torch.utils.data.DataLoader(trainingset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    trainingsetLabelLoader = torch.utils.data.DataLoader(y, batch_size=batch_size,
                            shuffle=False, num_workers=2)


    criterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sigmoid = nn.Sigmoid()
    
    losses = {'train':[], 'validation':[], 'TraPerEpoch':[], 'ValPerEpoch':[]}
    smallestValidationLossSeenSoFar = 10000

    for epoch in range(epochs):  # loop over the dataset multiple times
        trainRunningLoss = 0.0
        epochRunningLoss = 0.0
        
        for i, (data, label) in enumerate(zip(trainingsetLoader, trainingsetLabelLoader)):
            # wrap the data and labels from each batch in Variables
            inputs = Variable(data.cuda()) #remove .cuda to not put model on GPU
            labels = Variable(label.cuda()).float() #remove .cuda to not put model on GPU

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = sigmoid(model(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics and update running loss
            trainRunningLoss += loss.data[0]
            epochRunningLoss += loss.data[0]

            if i % 500 == 0:  # 100 is a better value. I have 500 here just for this application
                losses['TraPerEpoch'].append(epochRunningLoss / 100)
                epochRunningLoss = 0.0

                valRunningLoss = 0.0
                for image_id in val_ids[:]:            
                    img = load_image(val_id_to_file[image_id])
                    
                    pred = sigmoid(model(img))

                    valRunningLoss += criterion(outputs, labels).data[0]
                
                thisRoundsLoss = valRunningLoss / len(val_ids)
                losses['ValPerEpoch'].append(thisRoundsLoss)

                print("epoch: ",epoch, ", batch: ", i,", avg Training loss ", loss.data[0], "  validation loss ", valRunningLoss)
                
                if thisRoundsLoss < smallestValidationLossSeenSoFar:
                    print("Saving best model: epoch: ",epoch, ", batch: ", i,"  previous best validation loss ", smallestValidationLossSeenSoFar, "  new best validation loss ", thisRoundsLoss)
                    
                    smallestValidationLossSeenSoFar = thisRoundsLoss
                    model.save_checkpoint({
                            'epoch': epoch,
                            'batch': i,
                            'state_dict': model.state_dict(),
                            'val_error': smallestValidationLossSeenSoFar,
                            'optimizer' : optimizer.state_dict(),
                        }, is_best=True
                    )

        losses['train'].append(trainRunningLoss / len(trainingsetLoader))

        valRunningLoss = 0.0
        for image_id in val_ids[:]:            
            img = load_image(val_id_to_file[image_id])
            
            pred = sigmoid(model(img))

            valRunningLoss += criterion(outputs, labels).data[0]
        
        losses['validation'].append(valRunningLoss / len(val_ids))


    print('Final Training losses:\n', losses['train'])
    print('Final Validation losses:\n', losses['validation'])
    return losses
################### END: Training method ######################




###############################################################
############## START: Loading Saved Models ####################
###############################################################
#example filename for script:'./losses-alp_0.0001_batch_25.pth.tar'
#example filename for jupyter: 'losses-alp_0.0001_batch_25.pth.tar'
def loadSavedModel(filename):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) (batch {}) (error {})"
              .format("checkpoint.pth.tar", checkpoint['epoch'], checkpoint['batch'], checkpoint['val_error']))
    else:
        print("reeeedim!!")
################ END: Loading Saved Models ####################


###############################################################
####################### START: Start  #########################
###############################################################
train(model)