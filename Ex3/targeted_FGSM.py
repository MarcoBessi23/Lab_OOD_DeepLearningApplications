import argparse
import os
import torch
import torch.utils.data
from torch.utils.data import Subset
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def clean_accuracy(model, test_loader, device):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for im, lab in test_loader:
            im, lab = im.to(device), lab.to(device)
            logit   = model(im)
            _, predicted = torch.max(F.softmax(logit, 1), 1)
            accuracy += (predicted == lab).sum().item() / len(lab)
    return accuracy / len(test_loader)


transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
torch.manual_seed(42)
subset_indices = torch.randperm(len(testset))[:1000]
subset_testset = Subset(testset, subset_indices)

classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
denormalize = transforms.Normalize((-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010), 
                                       (1 / 0.2023, 1 / 0.1994, 1 / 0.2010))

class FGSM_targeted():
    def __init__(self, target_class, model):
        
        self.target_class = target_class
        self.model  = model
        self.test_loader =  torch.utils.data.DataLoader(subset_testset, batch_size = 1, shuffle = False)  
        self.loss   = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.examples = []

    def run_attack(self, eps = 0.2):

        self.model.eval()
        self.target = classes.index(self.target_class)
        self.target = torch.tensor([self.target], device = self.device)        
        success = 0
        difference = 0
        untargeted_success = 0
        perturbed_examples = []
        num_data = len(self.test_loader)
        for im, lab in self.test_loader:
            im, lab = im.to(self.device), lab.to(self.device)
            im.requires_grad = True
            logit_clean = self.model(im)

            ###CONTROL IF THE MODEL HAS ALREADY CLASSIFIED DESIRED TARGET
            ###OR IT IS ALREADY WRONG, SO ATTACK ONLY THE CORRECTLY CLASSIFIED LABELS
            if lab.item() == self.target:
                num_data -= 1
                continue
            if lab.item() != torch.max(logit_clean, dim=1)[1]:
                num_data -=1
                continue
            l = self.loss(logit_clean, self.target)
            self.model.zero_grad()
            l.backward()
            with torch.no_grad():
                perturbed_im = im - eps * im.grad.sign()
                logit_perturbed = self.model(perturbed_im)
                _, predicted_adversarial = torch.max(F.softmax(logit_perturbed, 1), 1)
            if predicted_adversarial.item() == self.target:
                success += 1
                difference += torch.norm(im.detach()-perturbed_im.detach())
                if len(perturbed_examples) < 5:
                    self.examples.append((lab.item(),im, predicted_adversarial.item(), perturbed_im ) )
            elif predicted_adversarial.item() != lab.item():
                untargeted_success += 1

        print(f'num succesfull attacks {success}')
        return success, difference/success, 1-(success+untargeted_success)/num_data

    def visualize(self):
        result_dir = 'Ex3/Results'
        os.makedirs('Ex3/Results', exist_ok=True)
        path_images =  os.path.join(os.getcwd(), result_dir, 'Qualitative_Confront.png')
        img     = denormalize(self.examples[0][1]).squeeze().detach().cpu().numpy()
        img_adv = denormalize(self.examples[0][3]).squeeze().detach().cpu().numpy()
        diff    = denormalize(self.examples[0][3] - self.examples[0][1]).squeeze().detach().cpu().numpy()

        img     = np.transpose(img,     (1, 2, 0))
        img_adv = np.transpose(img_adv, (1, 2, 0))
        diff    = np.transpose(diff,    (1, 2, 0))
        _, ax = plt.subplots(1, 3, figsize = (10, 4))
        ax[0].imshow(img)
        ax[0].set_title(f'Original: {classes[self.examples[0][0]]}')
        ax[1].imshow(img_adv)
        ax[1].set_title(f'Perturbed: {classes[self.examples[0][2]]}')
        ax[2].imshow(diff)
        ax[2].set_title('Difference')
        
        for a in ax:
            a.axis('off')
        
        plt.savefig(path_images)

    def confront_eps(self, epsilon_list):
        result_dir = 'Ex3/Results'
        os.makedirs('Ex3/Results', exist_ok=True)
        path_confront_success =  os.path.join(os.getcwd(), result_dir, 'Quantitative_Confront.png')
        path_confront_differences = os.path.join(os.getcwd(), result_dir, 'Differences.png')
        path_confront_acc = os.path.join(os.getcwd(), result_dir, 'Acuracy.png')
        self.examples = []
        differences_L2 = []
        successfull_attacks = []
        accuracy = []
        for eps in epsilon_list:
            successes, diff, acc = self.run_attack(eps)
            successfull_attacks.append(successes)
            differences_L2.append(diff)
            accuracy.append(acc)
        
        plt.plot(epsilon_list, successfull_attacks)
        plt.xlabel('epsilon values')
        plt.ylabel('succesfull attacks')
        plt.title('attacks VS epsilon')
        plt.savefig(path_confront_success)
        plt.close()

        plt.plot(epsilon_list, differences_L2)
        plt.xlabel('epsilon values')
        plt.ylabel('L_2 difference norm')
        plt.title('differences VS epsilon')
        plt.savefig(path_confront_differences)
        plt.close()

        plt.plot(epsilon_list, accuracy)
        plt.xlabel('epsilon values')
        plt.ylabel('accuracy')
        plt.title('accuracy VS epsilon')
        plt.savefig(path_confront_acc)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="FGSM Attack on CIFAR-10")
    parser.add_argument("--mode", choices=["quantitative", "qualitative"], required=True,
                        help="Choose between 'quantitative' (confront_eps) or 'qualitative' (run_attack + visualize)")
    
    #I USED CAR FOR THE EXPERIMENT
    parser.add_argument("--target", type=str, default="car",
                        help="Target class for the FGSM attack")
    
    args = parser.parse_args()

    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet44', pretrained=True)
    fgsm = FGSM_targeted(args.target, model)

    if args.mode == "quantitative":
        epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
        fgsm.confront_eps(epsilons)
    elif args.mode == "qualitative":
        fgsm.run_attack()
        fgsm.visualize()

if __name__ == "__main__":
    main()
