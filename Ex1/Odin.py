## Exercise 3: Wildcard
### Exercise 3.1: Implement ODIN for OOD detection
# ODIN is a very simple approach, and you can already start experimenting 
# by implementing a temperature hyperparameter in your base model and doing a grid search on T and epsilon.

#### Exercise 3.3: Experiment with *targeted* adversarial attacks
#Implement the targeted Fast Gradient Sign Method to generate adversarial samples that *imitate* samples from a specific class. Evaluate your adversarial samples qualitatively and quantitatively.
import os
import torch
import torch.utils
import torchvision
from torchvision.datasets import FakeData
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn import metrics
from NeuralNet import CNN, heatmap
from torch.utils.data import Subset
from sklearn.metrics import roc_curve, auc


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) ])

batch_size = 200

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)

import random
subset_indices = random.sample(range(len(testset)), 100)
testset = Subset(testset, subset_indices)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

fakeset = FakeData(size=len(testset), image_size=(3, 32, 32), transform=transform)
fakeloader = torch.utils.data.DataLoader(fakeset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss   = nn.CrossEntropyLoss()
name_model = 'resnet44'

e = np.linspace(0, 0.05, 10)  #epsilon parameters
t = np.array([1, 5, 10, 50, 100, 500, 1000, 2000]) #temperature parameters


if name_model == 'CNN':
    model  = CNN().to(device)
    model.load_state_dict(torch.load('Ex1/model.pt',map_location=torch.device(device)))
elif name_model == 'resnet44':
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet44', pretrained=True)

model.to(device)


def max_softmax(logit, T=1.0):
    
    out = logit/T
    s = F.softmax(out, dim = 1)
    sm, _ = s.max(dim=1) #get the max for each element of the batch
    return sm

def input_gradient(input, loss, model, T):

    input.requires_grad = True
    output = model(input)
    
    nnOut = F.softmax(output, dim = 1)
    _, maxIndexTemp = torch.max(nnOut, dim = 1)
    labels = maxIndexTemp.to(device)
        
    output = output/T
    model.zero_grad()
    l = loss(output, labels)
    l.backward()
    gradient = -input.grad
    gradient = gradient.detach()
    
    return gradient
        

def OOD_det(model, loss,  data_loader, T, eps):
    score = []
    model.eval()
    for (i, data) in enumerate(data_loader):
        images, _ = data
        images = images.to(device)
        gradient = input_gradient(images,loss, model, T)

        v = images + eps*gradient.sign()
        with torch.no_grad():
            logit = model(v)
            score.append(max_softmax(logit, T))
    score_t = torch.cat(score)
    return score_t


def grid_search(epsilon, temperatures):

    roc_auc = np.zeros((len(epsilon), len(temperatures)))
    best_auc = 0
    best_temp = 0
    best_eps  = 0
    
    for i, eps in enumerate(epsilon):
        for j, temp in enumerate(temperatures):
            print(f'evaluating for eps = {eps}, temperature = {temp}')
            scores_fake = OOD_det(model, loss, fakeloader, temp, eps)
            scores_test = OOD_det(model, loss, testloader, temp, eps)
            y_test = torch.ones_like(scores_test)
            y_fake = torch.zeros_like(scores_fake)
            y      = torch.cat((y_test, y_fake))
            y_pred = torch.cat((scores_test, scores_fake))
            fpr, tpr, _ = metrics.roc_curve(y, y_pred)
            roc_auc[i][j] = metrics.auc(fpr, tpr)
            if roc_auc[i][j]>best_auc:
                best_auc = roc_auc[i][j]
                best_eps, best_temp  =  eps, temp
                   
            print(f"AUC for eps = {eps:.3f} , temp = {temp} is {roc_auc[i][j] :.4f}")
    return roc_auc, best_auc, best_temp, best_eps


def grid_representation(heat_matrix, epsilon, temperature, name_model):
    
    path_AUC = os.path.join(os.getcwd(), 'Ex1/Results', f'{name_model}_AUC.png')
    fig, ax = plt.subplots()
    im, cbar = heatmap(heat_matrix, epsilon, temperature, ax=ax,
                       cmap="hot", cbarlabel="AUC score")
    fig.tight_layout()
    plt.savefig(path_AUC)
    plt.close()


score = grid_search(e, t)
e     = np.round(e, 3)
grid_representation(score[0], e, t, name_model)


#path_AUC = os.path.join(os.getcwd(), 'Ex1/Results', 'AUC.png')
#fig, ax = plt.subplots()
#
#e_rounded = np.round(e, 3)
#im, cbar = heatmap(score[0], e_rounded, t, ax=ax,
#                   cmap="hot", cbarlabel="AUC score")
#fig.tight_layout()
#plt.savefig(path_AUC)
#plt.close


def ROC_AUCvsBaseline(model, loss, testloader, fakeloader, best_temp, best_eps, name_model ):
    
    scores_test = OOD_det(model, loss, testloader, best_temp, best_eps)
    scores_fake = OOD_det(model, loss, fakeloader, best_temp, best_eps)

    scores_baseline_test = OOD_det(model, loss, testloader, 1, 0)
    scores_baseline_fake = OOD_det(model, loss, fakeloader, 1, 0)

    y_test = torch.ones_like(scores_test)
    y_fake = torch.zeros_like(scores_fake)

    y = torch.cat((y_test, y_fake))
    y_pred = torch.cat((scores_test, scores_fake))
    y_pred_baseline = torch.cat((scores_baseline_test, scores_baseline_fake))


    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)

    fpr_base, tpr_base, thresholds = roc_curve(y, y_pred_baseline)
    baseline_auc = auc(fpr_base, tpr_base)

    path_ROC_curve = os.path.join(os.getcwd(), 'Ex1/Results', f'{name_model}_Roc_CURVE.png')

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot(fpr_base, tpr_base, color='navy', lw=2, label=f'baseline (area = {baseline_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(path_ROC_curve)
    plt.close()


ROC_AUCvsBaseline(model, loss, testloader, fakeloader, score[2], score[3], name_model)

#scores_test = OOD_det(model, loss, testloader, score[2], score[3])
#scores_fake = OOD_det(model, loss, fakeloader, score[2], score[3])
#
#scores_baseline_test = OOD_det(model, loss, testloader, 1, 0)
#scores_baseline_fake = OOD_det(model, loss, fakeloader, 1, 0)
#
#y_test = torch.ones_like(scores_test)
#y_fake = torch.zeros_like(scores_fake)
#
#y = torch.cat((y_test, y_fake))
#y_pred = torch.cat((scores_test, scores_fake))
#y_pred_baseline = torch.cat((scores_baseline_test, scores_baseline_fake))
#
#
#fpr, tpr, thresholds = roc_curve(y, y_pred)
#roc_auc = auc(fpr, tpr)
#
#fpr_base, tpr_base, thresholds = roc_curve(y, y_pred_baseline)
#baseline_auc = auc(fpr_base, tpr_base)
#
#path_ROC_curve = os.path.join(os.getcwd(), 'Ex1/Results', 'Roc_CURVE.png')
#
#plt.figure()
#plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#plt.plot(fpr_base, tpr_base, color='navy', lw=2, label=f'baseline (area = {baseline_auc:.2f})')
##plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC)')
#plt.legend(loc='lower right')
#plt.savefig(path_ROC_curve)
#plt.close()