### Exercise 3.2: Implement JARN
# In exercise 2.2 you already implemented Jacobian-regularized learning to make your model 
# more robust to adversarial samples.Add a *discriminator* to your model to encourage the 
# adversarial samples used for regularization to be more *salient*.
#See [the JARN paper](https://arxiv.org/abs/1912.10185) for more details.
#
import os
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn import metrics
from torch.utils.data import Subset
from Neural_net import CNN, Discriminator
from TrainJarn import Trainer, JARN_trainer

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=False, transform= transform)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=False, transform= transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False)

model = CNN()
model.to(device)
#model.load_state_dict(torch.load('Ex2/model_parameters/standard_CNN.pth', map_location = torch.device(device)))


#model = model.load_state_dict(torch.load('Ex2/model_parameters/standard_CNN.pth'))
#model.to(device)

#loss = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


#trainer = Trainer(model, loss, optimizer, 3, device)
#trainer.train_model(train_loader, test_loader)
#trainer.save_model_parameters()

def clean(model, test_loader):
    accuracy = 0
    model.to(device)
    model.eval()
    for i, data in enumerate(test_loader):
        im, lab = data
        im,lab = im.to(device), lab.to(device)
        with torch.no_grad():
            logit = model(im)
            _, predicted = torch.max(F.softmax(logit, 1), 1)
            accuracy += (predicted == lab).sum().item()/len(lab) #accuracy on the specific batch  
    accuracy  /= len(test_loader)
    return accuracy

def FGSM_attack(model, test_loader, eps):
    accuracy = 0
    model.train()
    for i, data in enumerate(test_loader):
        print(f'iteration number {i}')
        im, lab = data
        im, lab = im.to(device), lab.to(device)
        im.requires_grad = True
        logit_clean = model(im)
        model.zero_grad()
        l = loss(logit_clean, lab)
        l.backward()
        gradient = im.grad
        gradient = gradient.detach()
        perturbed_im = im + eps * gradient.sign()
        with torch.no_grad():
            
            logit_perturbed = model(perturbed_im)
            _, predicted = torch.max(F.softmax(logit_perturbed, 1), 1)
            accuracy += (predicted == lab).sum().item()/len(lab) #accuracy on the specific batch  
    accuracy  /= len(test_loader)
    return accuracy


#print(FGSM_attack(model, test_loader, eps = 0.4))

def PGD_perturbation(image, labels, model, loss, alpha, eps,  iter):
    original_image = image.clone().detach()
    x_max = original_image + eps
    x_min = original_image - eps
    x_adv = image.clone().detach()
    image = image.detach()
    for _ in range(iter):
        x_adv.requires_grad = True
        logit = model(x_adv)
        l     = loss(logit, labels)
        model.zero_grad()
        l.backward()
        gradient = x_adv.grad
        perturbed_image = x_adv + alpha*gradient.sign()
        x_adv = torch.clamp(perturbed_image, x_min, x_max).detach()
    return x_adv

def PGD_attack(model, loss, test_loader, alpha,eps, iter):
    accuracy = 0
    for i, data in enumerate(test_loader):
        print(f'iteration {i}')
        im, lab = data
        perturbed_im = PGD_perturbation(im, lab, model,loss, alpha, eps, iter)
        with torch.no_grad():
            logit_perturbed = model(perturbed_im)
            _, predicted = torch.max(F.softmax(logit_perturbed, 1), 1)
            accuracy += (predicted == lab).sum().item()/len(lab) #accuracy on the specific batch
    accuracy /= len(test_loader)
    return accuracy

#print(PGD_attack(model, loss, test_loader, alpha = 0.05, eps = 0.3 ,iter = 5))


#im, lab = next(iter(test_loader))
#perturbed = PGD_perturbation(im, lab, model,loss, alpha= 0.5, eps= 0.3, iter= 5)
#plotting(im[0], perturbed[0])

#_, predicted = torch.max(F.softmax(model(im), 1), 1)
#accuracy = (predicted == lab).sum().item()/len(lab)
#print(accuracy)
#
#_, predicted = torch.max(F.softmax(model(perturbed), 1), 1)
#accuracy = (predicted == lab).sum().item()/len(lab)
#print(accuracy)
#
model = CNN()
disc = Discriminator()
apt   = nn.Sequential(
        nn.Conv2d(1,16,3, padding = 1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16,1,1),
        nn.Tanh()
    )



opt_model = torch.optim.Adam(model.parameters(), lr=1e-3)
opt_disc = torch.optim.Adam(disc.parameters(), lr = 1e-3)
opt_apt = torch.optim.Adam(apt.parameters(), lr = 1e-3)

#JARN = JARN_trainer(model, disc, apt, loss, opt_model, opt_disc, opt_apt, epochs=8, lam_adv= 1, epsilon = 0.3, device= device)
#JARN.train_model(train_loader, test_loader, update_adv=10)
#JARN.save_model_parameters()

#model_jarn = CNN()
#model_jarn.load_state_dict(torch.load('Ex2/model_parameters/JARN_CNN.pth', map_location = torch.device(device)))
#print('Accuracy of model trained with jacobian regularizer')
##print(FGSM_attack(model_jarn, test_loader, eps = 0.3))
#print(PGD_attack(model_jarn, loss, test_loader, alpha = 0.3 ,eps = 0.3, iter = 5))
#
#
#model_standard = CNN()
#model_standard.load_state_dict(torch.load('Ex2/model_parameters/standard_CNN.pth', map_location = torch.device(device)))
#print('accuracy of standard model')
##print(FGSM_attack(model_standard, test_loader, eps = 0.3))
#print(PGD_attack(model_standard, loss, test_loader, alpha = 0.3 ,eps = 0.3, iter = 5))




import matplotlib.pyplot as plt
def plotting(original, fgsm_adv, pgd_adv, path):

    f, ax = plt.subplots(1,3, figsize=(10, 4))
    ax[0].imshow(original.detach().numpy().reshape(28, 28), cmap='gray')
    ax[0].set_title("Original")
    ax[1].imshow(fgsm_adv.detach().numpy().reshape(28, 28), cmap='gray')
    ax[1].set_title("FGSM Perturbed")
    ax[2].imshow(pgd_adv.detach().numpy().reshape(28, 28), cmap='gray')
    ax[2].set_title("PGD Perturbed")
    
    for a in ax:
        a.axis('off')

    plt.savefig(path)
    plt.close()

model_standard = CNN()
model_standard.load_state_dict(torch.load('Ex2/model_parameters/standard_CNN.pth', map_location = torch.device(device)))
loss = nn.CrossEntropyLoss()

im, lab = next(iter(test_loader))
path_fgsm = os.path.join(os.getcwd(), 'Ex2/images', 'fgsm_perturbed.png')
path_pgd  = os.path.join(os.getcwd(), 'Ex2/images', 'pgd_perturbed.png')
path_perturbed  = os.path.join(os.getcwd(), 'Ex2/images', 'perturbed.png')
pgd_perturbed = PGD_perturbation(im, lab, model,loss, alpha= 0.3, eps= 0.3, iter= 5)


def fgsm_perturbation(im, lab,  model, loss, eps):
    im.requires_grad = True
    logit_clean = model(im)
    model.zero_grad()
    l = loss(logit_clean, lab)
    l.backward()
    gradient = im.grad
    gradient = gradient.detach()
    perturbed_im = im + eps * gradient.sign()

    return perturbed_im


fgsm_perturbed = fgsm_perturbation(im, lab, model_standard, loss, eps = 0.3)

plotting(im[0], fgsm_perturbed[0],pgd_perturbed[0], path_perturbed)