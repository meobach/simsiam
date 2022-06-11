# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import LRScheduler, ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
import ignite.contrib.engines.common as common

import opendatasets as od
import os
from random import randint
import urllib
import zipfile

# Define device to use (CPU or GPU). CUDA = GPU support for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
DATA_DIR = 'tiny-imagenet-200' # Original images come in shapes of [3,64,64]

# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')
# Functions to display single or a batch of sample images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def show_batch(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()    
    imshow(make_grid(images)) # Using Torchvision.utils make_grid function
    
def show_image(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    random_num = randint(0, len(images)-1)
    imshow(images[random_num])
    label = labels[random_num]
    print(f'Label: {label}, Shape: {images[random_num].shape}')

# Setup function to create dataloaders for image datasets
def generate_dataloader(data, name, transform):
    if data is None: 
        return None
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    
    # Wrap image dataset (defined above) in dataloader 
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=(name=="train"), 
                        **kwargs)
    
    return dataloader

# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file
val_img_dir = os.path.join(VALID_DIR, 'images')

# Open and read val annotations text file
fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
data = fp.readlines()

# Create dictionary to store img filename (word 0) and corresponding
# label (word 1) for every line in the txt file (as key value pair)
val_img_dict = {}
for line in data:
    words = line.split('\t')
    val_img_dict[words[0]] = words[1]
fp.close()

# Display first 10 entries of resulting val_img_dict dictionary
{k: val_img_dict[k] for k in list(val_img_dict)[:10]}

# Create subfolders (if not present) for validation images based on label,
# and move images into the respective folders
for img, folder in val_img_dict.items():
    newpath = (os.path.join(val_img_dir, folder))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

# Define transformation sequence for image pre-processing
# If not using pre-trained model, normalize with 0.5, 0.5, 0.5 (mean and SD)
# If using pre-trained ImageNet, normalize with mean=[0.485, 0.456, 0.406], 
# std=[0.229, 0.224, 0.225])

preprocess_transform_pretrain = T.Compose([
                T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])


# Define batch size for DataLoaders
batch_size = 512

# Create DataLoaders for pre-trained models (normalized based on specific requirements)
train_loader_pretrain = generate_dataloader(TRAIN_DIR, "train",
                                  transform=preprocess_transform_pretrain)

val_loader_pretrain = generate_dataloader(val_img_dir, "val",
                                 transform=preprocess_transform_pretrain)


import torchvision.models as models
#Define model architecture (using efficientnet-b3 version)
model_ft = models.resnet18(pretrained=True)

#load pretrained model
checkpoint = torch.load("checkpoint_0100.pth.tar", map_location="cpu")
# rename moco pre-trained keys
state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):
    # retain only encoder up to before the embedding layer
    if k.startswith('encoder') and not k.startswith('encoder.fc'):
        # remove prefix
        state_dict[k[len("encoder."):]] = state_dict[k]
    # delete renamed or unused k
    del state_dict[k]
#print(state_dict.keys())
#print("break")
state_dict_model=dict(model_ft.state_dict())
for key in state_dict.keys():
    state_dict_model[key]=state_dict[key]
#print(state_dict_model.keys())
model_ft.load_state_dict(state_dict_model)
# #model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)
for name, param in model_ft.named_parameters():
    #print(name)
    if "fc" not in name:
        param.requires_grad = False
#     # init the fc layer
#for name, param in model_ft.named_parameters():
    #print(name,param.requires_grad)
model_ft.fc.weight.data.normal_(mean=0.0, std=0.01)
model_ft.fc.bias.data.zero_()
# Move model to designated device (Use GPU when on Colab)
model = model_ft.to(device)


# Define hyperparameters and settings
#lr = 0.0001  # Learning rate
lr=0.1
num_epochs = 20  # Number of epochs
log_interval = 300  # Number of iterations before logging

# Set loss function (categorical Cross Entropy Loss)
loss_func = nn.CrossEntropyLoss()

# Set optimizer (using Adam as default)
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr,
#                                 momentum=0.9,
#                                 weight_decay=0.)


# Setup pytorch-ignite trainer engine
trainer = create_supervised_trainer(model, optimizer, loss_func, device=device)

# Add progress bar to monitor model training
ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"Batch Loss": x})

# Define evaluation metrics
metrics = {
    "accuracy": Accuracy(), 
    "loss": Loss(loss_func),
}

from ignite.metrics import Precision, Recall

precision = Precision(average=False)
recall = Recall(average=False)
# F1 = (precision * recall * 2 / (precision + recall)).mean()
# F1.attach(engine, "F1")

# Evaluator for training data
train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

# Evaluator for validation data
evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

# Display message to indicate start of training
@trainer.on(Events.STARTED)
def start_message():
    print("Begin training")

# Log results from every batch
@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_batch(trainer):
    batch = (trainer.state.iteration - 1) % trainer.state.epoch_length + 1
    print(f"Epoch {trainer.state.epoch} / {num_epochs}, "
          f"Batch {batch} / {trainer.state.epoch_length}: "
          f"Loss: {trainer.state.output:.3f}")

# Evaluate and print training set metrics
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(trainer):
    print(f"Epoch [{trainer.state.epoch}] - Loss: {trainer.state.output:.2f}")
    train_evaluator.run(train_loader_pretrain)
    epoch = trainer.state.epoch
    metrics = train_evaluator.state.metrics
    print(f"Train - Loss: {metrics['loss']:.3f}, "
          f"Accuracy: {metrics['accuracy']:.3f} ")

# Evaluate and print validation set metrics
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_loss(trainer):
    evaluator.run(val_loader_pretrain)
    epoch = trainer.state.epoch
    metrics = evaluator.state.metrics
    print(f"Validation - Loss: {metrics['loss']:.3f}, "
          f"Accuracy: {metrics['accuracy']:.3f}")

# Sets up checkpoint handler to save best n model(s) based on validation accuracy metric
common.save_best_model_by_val_score(
          output_path="best_models",
          evaluator=evaluator, model=model,
          metric_name="accuracy", n_saved=1,
          trainer=trainer, tag="val")

# Start training
trainer.run(train_loader_pretrain, max_epochs=num_epochs)

print(evaluator.state.metrics)