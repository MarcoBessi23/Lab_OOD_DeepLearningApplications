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
            _, predicted = torch.max(outputs, 1)
            #total   += labels.size(0)
            train_acc += (predicted == labels).sum().item()/len(labels) #accuracy on the specific batch


        #Divide for the batch size
        running_loss /= len(training_loader)
        train_acc    /= len(training_loader)

        return running_loss, train_acc 
            
    def evaluate_model(self, test_loader):
        
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




class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size= 3, dilation=1, padding = 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size= 3, dilation=1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size= 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size= 5, dilation=1, padding = 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size= 5, dilation=1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(3200,64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x