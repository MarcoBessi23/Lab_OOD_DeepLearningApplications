import torch.nn.functional as F
from torch import nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import os



class Trainer():
    def __init__(self, model, loss, optimizer, epochs, device = None):
        self.model = model
        self.loss  = loss
        self.optimizer = optimizer
        self.epochs    = epochs
        self.results = {'training_loss':[], 
                        'training_accuracy':[],                    
                        'test_loss':[],
                        'test_accuracy':[]}
        self.device = device


    def train_epoch(self, training_loader):
        
        self.model.train()
        running_loss = 0
        train_acc = 0
        for i, data in enumerate(training_loader):
            print(f'iteration {i+1} of {len(training_loader)}')
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            l = self.loss(outputs, labels)
            l.backward()
            self.optimizer.step()
            running_loss += l.item()

            #CALCULATE TRAIN ACCURACY
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                #total   += labels.size(0)
                train_acc += (predicted == labels).sum().item()/len(labels) #accuracy on the specific batch


        #Divide for the batch size
        running_loss /= len(training_loader)
        train_acc    /= len(training_loader)

        return running_loss, train_acc 

    def evaluate_model(self, test_loader):
        '''
        method to evaluate model performances with accuracy metric
        '''
        self.model.eval()
        test_loss = 0
        accuracy = 0

        for i, data in enumerate(test_loader):
            inputs, labels = data
            outputs = self.model(inputs)
            l = self.loss(outputs, labels)
            test_loss += l.item()
            
            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == labels).sum().item()/len(labels) #accuracy on the specific batch
        
        accuracy  /= len(test_loader)
        test_loss /= len(test_loader)

        return test_loss, accuracy


    def train_model(self, training_loader, test_loader):

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 10, gamma=0.1)
        for epoch in range(self.epochs):
            print(f'----------------------------EPOCH NUMBER {epoch}-------------------------')
            train_loss, train_acc = self.train_epoch(training_loader)
            test_loss , test_acc  = self.evaluate_model(test_loader)
            scheduler.step()
            print(f':::::::train_loss = {train_loss}:::::::')
            print(f':::::::train_acc = {train_acc}:::::::')
            print(f':::::::test_loss = {test_loss}:::::::')
            print(f':::::::test_acc = {test_acc} :::::::')
            self.results['training_loss'].append(train_loss)
            self.results['training_accuracy'].append(train_acc)
            self.results['test_loss'].append(test_loss)
            self.results['test_accuracy'].append(test_acc)
        return self.results

    def save_model_parameters(self):
        path = os.path.join(os.getcwd(), 'Ex2/model_parameters','standard_CNN.pth')
        torch.save(self.model.state_dict(), path)

    def plot_results(self):
        path_train = os.path.join(os.getcwd(), 'Ex2/results', 'loss.png')
        plt.plot(self.results['training_loss'], color = 'blue')
        plt.plot(self.results['test_loss'], color = 'red')
        plt.xlabel('epochs')
        plt.ylabel('training loss')
        plt.savefig(path_train)
        plt.close()

        path_accuracy = os.path.join(os.getcwd(), 'Ex2/results', 'accuracy.png')
        plt.plot(self.results['training_accuracy'], color= 'blue')
        plt.plot(self.results['test_accuracy'], color = 'red')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.savefig(path_accuracy)
        plt.close()


class JARN_trainer():
    def __init__(self, model, discriminator, apt, loss, optimizer_model , optimizer_disc, optimizer_apt, epochs, lam_adv, epsilon, device = None):
        self.model = model
        self.disc = discriminator
        self.apt = apt
        self.loss = loss
        self.opt_model = optimizer_model
        self.opt_apt   = optimizer_apt
        self.opt_disc  = optimizer_disc
        self.lam_adv = lam_adv
        self.eps = epsilon
        self.epochs = epochs
        self.results = {'training_loss':[], 
                        'training_accuracy':[],                    
                        'test_loss':[],
                        'test_accuracy':[]}
        self.device = device

    def prepare_disc(self, im, jac):
        disc_jac  = self.disc(jac)
        disc_real = self.disc(im)
        discrim   = torch.cat([disc_real, disc_jac])
        real_lab  = torch.ones_like(disc_real)
        fake_lab  = torch.zeros_like(disc_jac)
        adv_lab   = torch.cat([real_lab, fake_lab])
    
        return discrim, adv_lab

    def train_epoch(self, train_loader, update_adv):
        self.model.to(self.device)
        self.apt.to(self.device)
        self.disc.to(self.device)
        self.model.train()
        self.apt.train()
        self.disc.train()
        train_acc = 0
        running_loss = 0
        BCE = nn.BCELoss()
        
        for i, data in enumerate(train_loader):
            print(f'iteration number: {i}')
            image, label = data
            image, label = image.to(self.device), label.to(self.device)
            x = image.clone().detach()
            x.requires_grad = True
            logit = self.model(x)
            l_cls = self.loss(logit, label)
            self.model.zero_grad()
            l_cls.backward(retain_graph = True)

            jacobian = x.grad
            adjusted_jacobian = self.apt(jacobian)
            discrim, adv_lab = self.prepare_disc(image, adjusted_jacobian)
            
            l_adv = BCE(discrim, adv_lab)
            l = l_cls + self.lam_adv * l_adv
            
            #UPDATE MODEL PARAMETERS
            self.model.zero_grad()
            l.backward()
            self.opt_model.step()

            #UPDATE APT PARAMETERS
            discrim, adv_lab = self.prepare_disc(image, adjusted_jacobian.detach())
            l_adv = BCE(discrim, adv_lab)
            self.apt.zero_grad()
            l_adv.backward()
            self.opt_apt.step()

            if i%update_adv==0:
                #UPDATE DISCRIMINATOR PARAMETERS
                discrim, adv_lab = self.prepare_disc(image, adjusted_jacobian.detach())
                l_adv = BCE(discrim, adv_lab)
                self.disc.zero_grad()
                l_disc = -l_adv
                l_disc.backward()
                self.opt_disc.step()

            #CALCULATE TRAIN ACCURACY
            running_loss += l.item()
            
            with torch.no_grad():
                logit = self.model(image)
                _, prediction = torch.max(F.softmax(logit, 1),1)
                train_acc += (prediction == label).sum().item()/len(label) #accuracy on the specific batch

        #Divide for the batch size
        running_loss /= len(train_loader)
        train_acc    /= len(train_loader)

        return running_loss, train_acc 

     
    def evaluate_model(self, test_loader):
        accuracy = 0
        self.model.eval()
        for i, data in enumerate(test_loader):
            im, lab = data
            im, lab = im.to(self.device), lab.to(self.device)
            logit = self.model(im)
            _, prediction = torch.max(F.softmax(logit, 1),1)
            accuracy += (prediction==lab).sum().item()/len(lab)
        accuracy /= len(test_loader)
        
        return accuracy


    def train_model(self, train_loader, test_loader, update_adv):
        
        for epoch in range(self.epochs):
            print(f'----------------------------EPOCH NUMBER {epoch}-------------------------')
            train_loss, train_acc = self.train_epoch(train_loader, update_adv)
            test_acc  = self.evaluate_model(test_loader)
            print(f':::::::train_loss = {train_loss}:::::::')
            print(f':::::::train_acc = {train_acc}:::::::')
            print(f':::::::test_acc = {test_acc} :::::::')
            self.results['training_loss'].append(train_loss)
            self.results['training_accuracy'].append(train_acc)
            self.results['test_accuracy'].append(test_acc)
        return self.results
    
    def save_model_parameters(self):
        path = os.path.join(os.getcwd(), 'Ex2/model_parameters','JARN_CNN.pth')
        torch.save(self.model.state_dict(), path)