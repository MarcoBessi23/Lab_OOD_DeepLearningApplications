import argparse
import os
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
from Neural_net import CNN, Discriminator
from TrainJarn import Trainer, JARN_trainer

def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def clean_accuracy(model, test_loader, device):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for im, lab in test_loader:
            im, lab = im.to(device), lab.to(device)
            logit = model(im)
            _, predicted = torch.max(F.softmax(logit, 1), 1)
            accuracy += (predicted == lab).sum().item() / len(lab)
    return accuracy / len(test_loader)

def FGSM_attack(model, test_loader, eps, loss, device):
    model.train()
    accuracy = 0
    for im, lab in test_loader:
        im, lab = im.to(device), lab.to(device)
        im.requires_grad = True
        logit_clean = model(im)
        loss(logit_clean, lab).backward()
        perturbed_im = im + eps * im.grad.sign()
        with torch.no_grad():
            logit_perturbed = model(perturbed_im)
            _, predicted = torch.max(F.softmax(logit_perturbed, 1), 1)
            accuracy += (predicted == lab).sum().item() / len(lab)
    return accuracy / len(test_loader)

def PGD_attack(model, test_loader, loss, alpha, eps, num_iter, device):
    def PGD_perturbation(image, labels):
        x_adv = image.clone().detach()
        x_min, x_max = image - eps, image + eps
        for _ in range(num_iter):
            x_adv.requires_grad = True
            loss(model(x_adv), labels).backward()
            x_adv = torch.clamp(x_adv + alpha * x_adv.grad.sign(), x_min, x_max).detach()
        return x_adv
    
    accuracy = 0
    for im, lab in test_loader:
        im, lab = im.to(device), lab.to(device)
        perturbed_im = PGD_perturbation(im, lab)
        with torch.no_grad():
            logit_perturbed = model(perturbed_im)
            _, predicted = torch.max(F.softmax(logit_perturbed, 1), 1)
            accuracy += (predicted == lab).sum().item() / len(lab)
    return accuracy / len(test_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Choose mode: train or test')
    parser.add_argument('--train', choices=['standard', 'jarn'], help='Choose training type')
    parser.add_argument('--test_model', choices=['standard', 'jarn'], help='Choose model type for testing')
    parser.add_argument('--test', choices=['clean', 'fgsm', 'pgd'], help='Choose test type')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders()
    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    if args.mode == 'train':
        if args.train == 'jarn':
            disc = Discriminator()
            apt = nn.Sequential(
                nn.Conv2d(1,16,3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16,1,1), nn.Tanh()
            )
            opt_model = torch.optim.Adam(model.parameters(), lr=1e-3)
            opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-3)
            opt_apt = torch.optim.Adam(apt.parameters(), lr=1e-3)
            trainer = JARN_trainer(model, disc, apt, loss_fn, opt_model, opt_disc, opt_apt, 8, 1, 0.3, device)
        else:
            opt_model = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = Trainer(model, loss_fn, opt_model, 3, device)
        
        trainer.train_model(train_loader, test_loader)
        trainer.save_model_parameters()
    
    elif args.mode == 'test':
        if args.test_model == 'jarn':
            model.load_state_dict(torch.load('Ex2/model_parameters/JARN_CNN.pth', map_location=device))
        else:
            model.load_state_dict(torch.load('Ex2/model_parameters/standard_CNN.pth', map_location=device))
        
        if args.test == 'clean':
            print(f'Clean Accuracy: {clean_accuracy(model, test_loader, device)}')
        elif args.test == 'fgsm':
            print(f'FGSM Accuracy: {FGSM_attack(model, test_loader, 0.3, loss_fn, device)}')
        elif args.test == 'pgd':
            print(f'PGD Accuracy: {PGD_attack(model, test_loader, loss_fn, 0.3, 0.3, 5, device)}')

if __name__ == '__main__':
    main()
