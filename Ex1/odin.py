import os
import random
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FakeData
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from NeuralNet import CNN, heatmap
from sklearn import metrics


class OODDetector:
    def __init__(self, model_name='resnet44', batch_size=200, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.loss = nn.CrossEntropyLoss()
        self.model_name = model_name
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.testloader, self.fakeloader = self.load_data()
        self.model = self.load_model()

    def load_data(self):
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=self.transform)
        fakeset = FakeData(size=len(testset), image_size=(3, 32, 32), transform=self.transform)
        return (DataLoader(testset, batch_size=self.batch_size, shuffle=False),
                DataLoader(fakeset, batch_size=self.batch_size, shuffle=False))

    def load_model(self):
        if self.model_name == 'CNN':
            model = CNN().to(self.device)
            model.load_state_dict(torch.load('Ex1/model.pt', map_location=self.device))
        elif self.model_name == 'resnet44':
            model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet44', pretrained=True)
        return model.to(self.device)

    def max_softmax(self, logit, T=1.0):
        return F.softmax(logit / T, dim=1).max(dim=1)[0]

    def input_gradient(self, input, T):
        input.requires_grad = True
        output = self.model(input) / T
        labels = output.argmax(dim=1).to(self.device)
        self.model.zero_grad()
        loss = self.loss(output, labels)
        loss.backward()
        return -input.grad.detach()

    def OOD_det(self, data_loader, T, eps):
        self.model.eval()
        scores = []
        for images, _ in data_loader:
            images = images.to(self.device)
            gradient = self.input_gradient(images, T)
            perturbed_images = images + eps * gradient.sign()
            with torch.no_grad():
                logit = self.model(perturbed_images)
                scores.append(self.max_softmax(logit, T))
        return torch.cat(scores)

    def grid_search(self, epsilon, temperatures):
        best_auc, best_eps, best_temp = 0, 0, 0
        roc_auc = np.zeros((len(epsilon), len(temperatures)))
        for i, eps in enumerate(epsilon):
            for j, temp in enumerate(temperatures):
                print(f'Evaluating: eps={eps}, temp={temp}')
                scores_fake = self.OOD_det(self.fakeloader, temp, eps)
                scores_test = self.OOD_det(self.testloader, temp, eps)
                y = torch.cat((torch.ones_like(scores_test), torch.zeros_like(scores_fake)))
                y_pred = torch.cat((scores_test, scores_fake))
                fpr, tpr, _ = metrics.roc_curve(y.cpu().numpy(), y_pred.cpu().numpy())
                roc_auc[i, j] = metrics.auc(fpr, tpr)
                if roc_auc[i, j] > best_auc:
                    best_auc, best_eps, best_temp = roc_auc[i, j], eps, temp
                print(f'AUC: {roc_auc[i, j]:.4f}')
        return roc_auc, best_auc, best_temp, best_eps

    def grid_representation(self, heat_matrix, epsilon, temperature):
        path_AUC = os.path.join(os.getcwd(), 'Ex1/Results', f'{self.model_name}_AUC.png')
        fig, ax = plt.subplots()
        heatmap(heat_matrix, epsilon, temperature, ax=ax, cmap="hot", cbarlabel="AUC score")
        plt.savefig(path_AUC)
        plt.close()

    def ROC_AUCvsBaseline(self, best_temp, best_eps):
        path_ROC_curve = os.path.join(os.getcwd(), 'Ex1/Results', f'{self.model_name}_ROC_CURVE.png')
        scores_test = self.OOD_det(self.testloader, best_temp, best_eps)
        scores_fake = self.OOD_det(self.fakeloader, best_temp, best_eps)
        scores_baseline_test = self.OOD_det(self.testloader, 1, 0)
        scores_baseline_fake = self.OOD_det(self.fakeloader, 1, 0)
        
        y = torch.cat((torch.ones_like(scores_test), torch.zeros_like(scores_fake)))
        y_pred = torch.cat((scores_test, scores_fake))
        y_pred_baseline = torch.cat((scores_baseline_test, scores_baseline_fake))
        
        fpr, tpr, _ = roc_curve(y.cpu().numpy(), y_pred.cpu().numpy())
        roc_auc_score = auc(fpr, tpr)
        fpr_base, tpr_base, _ = roc_curve(y.cpu().numpy(), y_pred_baseline.cpu().numpy())
        baseline_auc = auc(fpr_base, tpr_base)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC={roc_auc_score:.2f})')
        plt.plot(fpr_base, tpr_base, color='navy', lw=2, label=f'Baseline (AUC={baseline_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig(path_ROC_curve)
        plt.close()


if __name__ == "__main__":
    detector = OODDetector(model_name='resnet44')
    epsilons = np.linspace(0, 0.05, 10)
    temperatures = np.array([1, 5, 10, 50, 100, 500, 1000, 2000])
    score = detector.grid_search(epsilons, temperatures)
    detector.grid_representation(score[0], np.round(epsilons, 3), temperatures)
    detector.ROC_AUCvsBaseline(score[2], score[3])
