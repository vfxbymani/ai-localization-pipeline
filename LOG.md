# Learning Log

## Day 1: PyTorch basics
**What I built:**
- Tensors, tensor math, and autograd
- A linear model that learned y = 3x + 2 (weight 3.00, bias 2.03)
- A small CNN on FashionMNIST: 88.7% test accuracy after 3 epochs

**What I learned:**
- (one sentence about the training loop: guess, measure error, find slopes, nudge)
- (one sentence about what a CNN does)

**What confused me:**
- (one thing that was unclear, for example why the loss printed went up and down)

**Tools:** Kaggle free GPU, GitHub

## Day 2: Transfer learning
**What I built:**
- Loaded a pretrained ResNet18, froze it, and replaced the last layer (1,026 trainable parameters out of 11.2M)
- Ants vs bees, 244 training images: 93.5% peak validation accuracy (89.5% at epoch 5)
- Same model trained from scratch: 61.4% after 5 epochs

**What I learned:**
- (one sentence on why the pretrained model did so much better)
- (one sentence on what freezing layers does)

**What confused me:**
- (one thing, for example why validation accuracy moved up and down between epochs)

**Tools:** Kaggle free GPU, torchvision, GitHub
