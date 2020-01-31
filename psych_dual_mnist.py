# CS 394N Neural Networks
# Final Project
# Elizabeth Liner


# Based on https://github.com/yhuag/neural-network-lab/blob/master/Feedforward%20Neural%20Network.ipynb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time
import csv
import numpy
import random

def getRandom(min, max):
    return random.randint(min, max)


# Initialize variables
input_size = 784        # Image size = 28x28 = 784
hidden_size = 500       # Hidden nodes
num_classes = 10        # Output classes - 0-9
num_epochs = 10         # Number of times we train on the dataset
batch_size = 10          # Size of input data for a batch
correct_learning_rate = 0.05            # Speed of convergence
cor_lr_change = correct_learning_rate * 0. #0.000125   # Rate change
incorrect_learning_rate = 0.05          # Speed of convergence
incor_lr_change = incorrect_learning_rate * 0 #0.02 # Rate change

# Download MNIST dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

# Load datasets into batches
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

batch_acc_output = []
test_batch_acc_output = []
epoch_acc_output = []
dev_acc_output = []
cor_lr_output = []
incor_lr_output = []
dev_acc_output = []

# Feedforward Neural Network Model
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Create model
model = FFNN(input_size, hidden_size, num_classes)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=correct_learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=correct_learning_rate)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=correct_learning_rate)
optimizer = torch.optim.Adagrad(model.parameters(), lr=correct_learning_rate)

start_time = time.time()
ndevset = 0 #5000
nsamples = 1000 # - ndevset
print_ex = 100
save_ex = 100
min_rate = 195
max_rate = 205

# Training
for epoch in range(0, num_epochs):
    train_correct = 0.
    train_total = 0.
    dev_correct = 0.
    dev_total = 0.
    correct_count = 0
    incorrect_count = 0
    correct_rand_ratio = getRandom(min_rate, max_rate)
    incorrect_rand_ratio = getRandom(min_rate, max_rate)
    for i, (images, labels) in enumerate(train_loader):
        #if i > (nsamples + ndevset):
        #    break
        """
        if i > nsamples:
            # Test out dev set
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            dev_total += labels.size(0)
            dev_correct += (predicted == labels).sum().item()

            if i % 1000 == 0:
                print("Epoch [%d/%d], Step [%d/%d], Calculating Dev Accuracy..."
                    % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size))

            continue
        """

        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        
        # Checking for our extra reinforcement
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        current_correct = (predicted == labels).sum()
        current_incorrect = batch_size - current_correct
        train_correct += (predicted == labels).sum()


        # Alright so what's the idea.
        #   1. Dual learning rates - one for correct answers and one for incorrect answers
        #   2. The learning rates are updated as we get more and more correct and incorrect answers
        #       a. The "correct" learning rate is decreased as time goes on
        #       b. The "incorrect" learning rate is increased as time goes on
        #      I'm thinking about 0.1%, or even less. We'll play with it depending on the original lr.
        #
        # So how does it look?
        #   Run it through, figure out correctness.
        #   Determine if we need to update the learning rate and update it (getRandom)
        #   Use the correct learning rate based on what the answer is.
        
        #"""
        ex_correct_or_incorrect = 0
        if current_correct >= current_incorrect:
            correct_count += current_correct
            incorrect_count += current_incorrect
            ex_correct_or_incorrect = 1
        else:
            correct_count += current_correct
            incorrect_count += current_incorrect
            ex_correct_or_incorrect = 0

        # Update our learning rate based on our correct and incorrect responses
        if correct_count >= correct_rand_ratio:
            correct_learning_rate = correct_learning_rate - cor_lr_change
            correct_count = 0
            correct_rand_ratio = getRandom(min_rate, max_rate)
        elif incorrect_count >= incorrect_rand_ratio:
            incorrect_learning_rate = incorrect_learning_rate + incor_lr_change
            incorrect_count = 0
            incorrect_rand_ratio = getRandom(min_rate, max_rate)

        for param_group in optimizer.param_groups:
            if ex_correct_or_incorrect:
                param_group['lr'] = correct_learning_rate
            else:
                param_group['lr'] = incorrect_learning_rate
        #"""

        optimizer.step()

        if (i+1) % save_ex == 0:

            # Grab test accuracy
            test_correct = 0.
            test_total = 0.
            for test_images, test_labels in test_loader:
                test_images = Variable(test_images.view(-1, 28*28))
                test_outputs = model(test_images)
                _, test_predicted = torch.max(test_outputs.data, 1)
                test_total += test_labels.size(0)
                test_correct += (test_predicted == test_labels).sum()

            test_correct = test_correct.item()
            
            b_acc = (100. * train_correct.item() / train_total)
            batch_acc_output.append(b_acc)
            test_acc = (100. * test_correct / test_total)
            test_batch_acc_output.append(test_acc)
            cor_lr_output.append(correct_learning_rate)
            incor_lr_output.append(incorrect_learning_rate)
        if (i+1) % print_ex == 0:
            print("Epoch [%d/%d], Step [%d/%d]" % (epoch+1, num_epochs,
                        i+1, len(train_dataset)//batch_size))

    e_acc = (100. * train_correct.item() / train_total)
    print("Accuracy of network for this batch: %.4f %%" % (e_acc))
    epoch_acc_output.append(e_acc)
    
    #dev_acc = (100. * dev_correct / dev_total)
    #print("Dev accuracy of network for this batch: %.4f %%" % (dev_acc))
    #dev_acc_output.append(dev_acc)

end_time = time.time()
print("Training acc per %d exs: " % print_ex, batch_acc_output)
print("Training acc per epoch: ", epoch_acc_output)
#print("Dev acc per epoch:", dev_acc_output)
print("Training time is", (end_time - start_time))
print("Final Correct Learning rate is", correct_learning_rate)
print("Final Incorrect Learning rate is", incorrect_learning_rate)

# Testing
correct = 0.
total = 0.
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

correct = correct.item()
print("Accuracy of the network on the 10K test images: %.4f %%" % (100. * correct / total))

batch_print = numpy.asarray(batch_acc_output)
numpy.savetxt("b_out.csv", batch_print, delimiter=",")
test_batch_print = numpy.asarray(test_batch_acc_output)
numpy.savetxt("test_out.csv", test_batch_print, delimiter=",")
epoch_print = numpy.asarray(epoch_acc_output)
numpy.savetxt("e_out.csv", epoch_print, delimiter=",")
cor_lr_print = numpy.asarray(cor_lr_output)
numpy.savetxt("cor_lr_out.csv", cor_lr_print, delimiter=",")
incor_lr_print = numpy.asarray(incor_lr_output)
numpy.savetxt("incor_lr_out.csv", incor_lr_print, delimiter=",")
#dev_print = numpy.asarray(dev_acc_output)
#numpy.savetxt("d_out.csv", dev_print, delimiter=",")

