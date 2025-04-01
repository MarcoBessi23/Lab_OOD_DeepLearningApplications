# Out-of-Distribution (OOD) Detection

## Introduction

## **ODIN hyperparameters**
<span style="font-size: 18px;">I implemented the [ODIN](https://arxiv.org/abs/1706.02690) method for the detection of out of distribution images, finding of this work is that applying temperature scaling and introducing small perturbations to the input helps distinguish the softmax score distributions of in-distribution and out-of-distribution images, making detection more effective.
For this lab, I used the ResNet-44 model pretrained on CIFAR-10, which can be downloaded via torch.hub.load. As the out-of-distribution dataset, I generated a dataset using the FakeData method from torchvision.datasets.
My objective was to study the hyperparameters temperature and epsilon of ODIN using a grid search. To evaluate the network's ability to distinguish out-of-distribution images, I measured the ROC AUC score for each couple of hyperparameters.
<br><br>

<span style="font-size: 18px;">Here, I present a heatmap representation of the grid with all the obtained results. As we can observe, there doesn't seem to be a specific hyperparameter dimension that significantly influences the AUC. A near-perfect ROC AUC is achieved around ε = 0.017 and temperature = 1000.
<br><br>

![gridAUC](Ex1/Results/AUC.png)
<br><br>

<span style="font-size: 18px;">It is also evident that this method outperforms the baseline approach by [Hendrycks and Gimpel](https://arxiv.org/abs/1610.02136) represented by the top-left square in the heatmap.
For clarity I also report the comparison between the two ROC curves.
<br><br>

![confront ROC](Ex1/Results/ROC_CURVE.png)

<br><br>


## **JARN Implementation for MNIST**

<span style="font-size: 18px;">In this lab, I implemented [JARN](https://arxiv.org/pdf/1912.10185), an adversarial training method that leverages the Jacobian of the loss function with respect to input data as false images. The model, a simple CNN, is trained to minimize the standard cross-entropy loss on MNIST while simultaneously generating adversarial images, the jacobian, capable of deceiving a discriminator network. This procedure aims at training classifiers to produce more meaningful and image-like input Jacobian matrices and this improve their resistance to adversarial examples.See the [original paper](https://arxiv.org/pdf/1912.10185) for a complete explanation. 

<span style="font-size: 18px;">To evaluate the effectiveness of JARN, I trained a CNN using both the 'standard' training method and JARN for a small number of training epochs, limiting the dataset to MNIST for computational simplicity, as I currently lack access to a GPU. The trained models were then tested against MNIST images perturbed using FGSM and PGD, to assess their resistance to adversarial perturbations, see the code for the details of the implementations.

<span style="font-size: 18px;">For illustration, here I show the effect of the attacks on an example image from the dataset:

<br><br>

![visualize attacks](Ex2/images/perturbed.png)

<br><br>

<span style="font-size: 18px;">As we can see, the performance of the standard network is dramatically degraded, dropping from 0.99 accuracy on the MNIST test set to 0.87 after an FGSM attack and 0.78 after PGD. In contrast, the network trained with JARN shows greater robustness, dropping only to 0.96 accuracy in the worst case.


<table style="width:50%; text-align:center; font-size:18px;">
  <tr>
    <th></th>
    <th>Clean</th>
    <th>FGSM</th>
    <th>PGD</th>
  </tr>
  <tr>
    <td><strong>Standard</strong></td>
    <td>0.99</td>
    <td>0.87</td>
    <td>0.78</td>
  </tr>
  <tr>
    <td><strong>Jarn</strong></td>
    <td>0.98</td>
    <td>0.97</td>
    <td>0.96</td>
  </tr>
</table>


## Usage Instructions

### Prerequisites
torch

### Running the Script
The script supports two main modes: **training** and **testing**.

#### 1. Training the Model
To train the model, use the following command:

```bash
python script.py --mode train --train <standard|jarn>
```

#### 2. Testing the Model
To test the trained model, use the following command, you can choose among the 3 different test:

```bash
python script.py --mode test --test_model <standard|jarn> --test <clean|fgsm|pgd>
```
## **Targeted FGSM attack**

<span style="font-size: 18px;">The Fast Gradient Sign Method (FGSM) is an adversarial attack technique that  modifies the inputs of a machine learning model in a way that tricks model into making an incorrect prediction. It does so by adding a small perturbation in the direction of the model’s gradients to the input image, maximizing the model's error. The Targeted version of FGSM that I developed in the following is a variant of this attack specifically designed to mislead the model into classifying an input as a target class chosen by the attacker. Results are reported both qualitatively by looking at the generated perturbed image that quantitatively studying the effects of perturbation magnitude parameters on some metrics, as shown below.

<br><br>

![visualize targeted attacks](Ex3/Results/Qualitative_Confront.png)

<br><br>
<span style="font-size: 18px;"> In the quantitative evaluation I started considering the model's accuracy on the perturbed images for different values of the epsilon parameter, as expected increasing the noise in the original image corresponds to a significative degradation of the performance. Never the less increasing the values of epsilon parameters comes at the cost of a greater 'distance' between the original and perturbed image, measured using the euclidean distance of the two images. It is interesting to note how the growth of the L2 norm can be used to explain the trend of the success rate curve of targeted adversarial attacks. In particular, when epsilon becomes too large, the perturbation may move the image out of the decision region where the model would classify it as the target class.

<br><br>
<p align="center">
  <img src="Ex3/Results/Acuracy.png" width="300" >
  <img src="Ex3/Results/Differences.png" width="300">
  <img src="Ex3/Results/Quantitative_Confront.png" width="300">
</p>

<br><br>

## Usage Instructions

### Prerequisites
torch

### Running the Script
The script supports two main modes: **quantitative** and **qualitative**.
To run the code just use the command below with one of the two modes as argument.

```bash
python targeted_FGSM.py --mode <quantitative|qualitative>
```