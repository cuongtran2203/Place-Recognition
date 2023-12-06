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
from metrics import ArcMarginProduct
def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name
if __name__=="__main__":
    device = torch.device("cuda")

    model=ConvTmixArcFace(model_pretrained="checkpoint_M.pth")
    '''
    Khoi tao arcface
    '''
    num_classes=1
    metric_fc = ArcMarginProduct(512,num_classes, s=30, m=0.5, easy_margin=False)
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)
    '''
    Khởi tạo loss,op
    
    '''
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=0.0001, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    criterion=torch.nn.CrossEntropyLoss()
    epochs=300
    start = time.time()
    for i in range(epochs):
        scheduler.step()
    trainloader=None
    model.train()
    for ii, data in enumerate(trainloader):
        data_input, label = data
        data_input = data_input.to(device)
        label = label.to(device).long()
        feature = model(data_input)
        output = metric_fc(feature, label)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iters = i * len(trainloader) + ii

        if iters % 100== 0:
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            # print(output)
            # print(label)
            acc = np.mean((output == label).astype(int))
            speed = 100/ (time.time() - start)
            time_str = time.asctime(time.localtime(time.time()))
            print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))

            start = time.time()

    if i % 100== 0 or i == epochs:
        save_model(model,"save_model","bestweights", i)