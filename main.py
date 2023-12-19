import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision
import torch
import numpy as np
import random
import time
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
import os
from convTmixarcface import ConvTmixArcFace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from metrics import ArcMarginProduct
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import Subset
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import GSVDataset
from dataset.transform import *
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/Deep_metriclearning_with_Tripletsloss')

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def train(model, loss_func, device, train_loader, optimizer, epoch):
    model.train()
    model.to(device)
    for batch_idx, (data, labels) in enumerate(train_loader):
        # print(data.shape)
        # print(labels)
        # data=torch.as_tensor(data)
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        # labels=labels.view(-1,1)
        # print("------------------",labels)
        
        # print(embeddings.shape)
        # print(labels.shape)
        
        loss = loss_func(embeddings, labels)

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss))
            writer.add_scalar("Loss train",len(train_loader) + batch_idx)
        if batch_idx%3000==0:
            print("change tesing mode")
            break


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

if __name__=="__main__":
    device = torch.device("cuda")

    model=ConvTmixArcFace(model_pretrained="checkpoint_M.pth")
    batch_size = 64

    dataset1 = GSVDataset(data_dir="./GG_datasets",transform=train_transform)
    datasets = train_val_dataset(dataset1)
    train_loader=datasets["train"]
    test_loader=datasets["val"]

    
    # train_loader = torch.utils.data.DataLoader(
    #     dataset1, batch_size=batch_size, shuffle=True
    # )
    # test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

 
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    # num_epochs = 2


    ### pytorch-metric-learning stuff ###
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.4, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.4, distance=distance, type_of_triplets="semihard"
    )
    # loss_func = losses.SubCenterArcFaceLoss(num_classes=16, embedding_size=512).to(device)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    ### pytorch-metric-learning stuff ###
    
    num_epochs=300
    """
    Loop train
    """
    for epoch in range(1, num_epochs + 1):
        train(model, loss_func, device, train_loader, optimizer, epoch)
        test(datasets["train"], datasets["val"], model, accuracy_calculator)
