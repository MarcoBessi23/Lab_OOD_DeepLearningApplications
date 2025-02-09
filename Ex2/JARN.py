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

loss = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


#trainer = Trainer(model, loss, optimizer, 3, device)
#trainer.train_model(train_loader, test_loader)
#trainer.save_model_parameters()

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
    image          = image.detach()
    for _ in range(iter):
        image.requires_grad = True
        logit = model(image)
        l     = loss(logit, labels)
        model.zero_grad()
        l.backward()
        gradient = image.grad
        perturbed_image = image + alpha*gradient.sign()
        image = torch.min(x_max, torch.max(x_min, perturbed_image)).detach()

        #proj_dir = torch.clamp(perturbed_image - original_image, min=-eps, max=eps) #direction in R(28x28) of the projection
        #image    = torch.clamp(original_image + proj_dir, min=0, max=1).detach_()      #projection operation
    return image

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

def plotting(original, adv):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(original.detach().numpy().reshape(28, 28), cmap='gray')
    axarr[1].imshow(adv.detach().numpy().reshape(28, 28), cmap='gray')
    plt.show()

im, lab = next(iter(test_loader))
perturbed = PGD_perturbation(im, lab, model,loss, alpha= 0.5, eps= 0.3, iter= 5)
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
apt = nn.Sequential(
        nn.Conv2d(1,1,1),
        nn.Tanh()
    )

opt_model = torch.optim.Adam(model.parameters(), lr=1e-3)
opt_disc = torch.optim.Adam(disc.parameters(), lr = 1e-3)
opt_apt = torch.optim.Adam(apt.parameters(), lr = 1e-3)

JARN = JARN_trainer(model, disc, apt, loss, opt_model, opt_disc, opt_apt, epochs=3, lam_adv= 1, epsilon = 0.02, device= device)
JARN.train_model(train_loader, test_loader, update_adv=10)
JARN.save_model_parameters()

#def prepare_disc(im, jac):
#    disc_jac  = disc(jac)
#    disc_real = disc(im)
#    discrim   = torch.cat([disc_real, disc_jac])
#    real_lab  = torch.ones_like(disc_real)
#    fake_lab  = torch.zeros_like(disc_jac)
#    adv_lab   = torch.cat([real_lab, fake_lab])
#    
#    return discrim, adv_lab
#
#
#lam_adv = 1
#image, labels = next(iter(train_loader))
#image.requires_grad = True
#logit = model(image)
#l_cls = loss(logit, labels)
#model.zero_grad()
#l_cls.backward(retain_graph = True)
#gradient = image.grad
#jacobian = apt(gradient)
#adjusted_jacobian = apt(jacobian)
#discrim, adv_lab = prepare_disc(image, adjusted_jacobian)

#disc_jac  = disc(adjusted_jacobian)
#disc_real = disc(image)
#discrim   = torch.cat([disc_real, disc_jac])
#real_lab  = torch.ones_like(disc_real)
#fake_lab  = torch.zeros_like(disc_jac)
#adv_lab   = torch.cat([real_lab, fake_lab])
#print(discrim)
#print(adv_lab)




#BCE = nn.BCELoss()
#l_adv = BCE(discrim, adv_lab)
#l = l_cls + lam_adv * l_adv
#
#model.zero_grad()
#l.backward()
#opt_model.step()

##UPDATE ADAPTOR
#disc_jac  = disc(adjusted_jacobian.detach())
#disc_real = disc(image)
#discrim   = torch.cat([disc_real, disc_jac])
#real_lab  = torch.ones_like(disc_real)
#fake_lab  = torch.zeros_like(disc_jac)
#adv_lab   = torch.cat([real_lab, fake_lab])

#discrim, adv_lab = prepare_disc(image, adjusted_jacobian.detach())
#l_adv = BCE(discrim, adv_lab)
#apt.zero_grad()
#l_adv.backward()
#opt_apt.step()


##UPDATE DISCRIMINATOR

#disc_jac  = disc(adjusted_jacobian.detach())
#disc_real = disc(image)
#discrim   = torch.cat([disc_real, disc_jac])
#real_lab  = torch.ones_like(disc_real)
#fake_lab  = torch.zeros_like(disc_jac)
#adv_lab   = torch.cat([real_lab, fake_lab])

#discrim, adv_lab = prepare_disc(image, adjusted_jacobian.detach())
#l_adv = BCE(discrim, adv_lab)
#disc.zero_grad()
#l_disc = -l_adv
#l_disc.backward()
#opt_disc.step()
