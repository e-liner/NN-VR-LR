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
batch_size = 1          # Size of input data for a batch
learning_rate = 0.001   # Speed of convergence
lr_change = learning_rate * 0.0001

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
epoch_acc_output = []
dev_acc_output = []
lr_output = []
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
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

start_time = time.time()
ndevset = 0 #5000
nsamples = 60000 # - ndevset
print_ex = 1000
save_ex = 1000
min_rate = 4
max_rate = 6

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
        if i > (nsamples + ndevset):
            break
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
        current_incorrect = 1 - current_correct # note - only works with batch size 1
        train_correct += (predicted == labels).sum()
        
        if current_correct.item():
            correct_count += 1
        elif current_incorrect.item():
            incorrect_count += 1

        # Update our learning rate based on our correct and incorrect responses
        if correct_count == correct_rand_ratio:
            learning_rate = learning_rate + lr_change

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            correct_count = 0

            correct_rand_ratio = getRandom(min_rate, max_rate)
        elif incorrect_count == incorrect_rand_ratio:
            learning_rate = learning_rate - lr_change

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            incorrect_count = 0

            incorrect_rand_ratio = getRandom(min_rate, max_rate)

        optimizer.step()

        if (i+1) % save_ex == 0:
            b_acc = (100. * train_correct.item() / train_total)
            batch_acc_output.append(b_acc)
            lr_output.append(learning_rate)
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
print("Final Learning rate is", learning_rate)

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
epoch_print = numpy.asarray(epoch_acc_output)
numpy.savetxt("e_out.csv", epoch_print, delimiter=",")
lr_print = numpy.asarray(lr_output)
numpy.savetxt("lr_out.csv", lr_print, delimiter=",")
#dev_print = numpy.asarray(dev_acc_output)
#numpy.savetxt("d_out.csv", dev_print, delimiter=",")

