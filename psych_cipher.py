# CS 394N Neural Networks
# Final Project
# Elizabeth Liner

# Based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import numpy

def getRandom(min, max):
    return random.randint(min, max)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

batch_size = 4

# Get data and transform it into tensors
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

model = CNN()

# Parameters
learning_rate = 0.001
lr_change = 0.001 * 0.001
momentum = 0.9
num_epochs = 10

batch_acc_output = []
epoch_acc_output = []
loss_output = []
print_ex = 1000
min_rate = 95
max_rate = 105

# Loss Function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
#optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

start_time = time.time()

# Train the Network
for epoch in range(0, num_epochs):
    train_correct = 0.
    train_total = 0.
    correct_count = 0
    incorrect_count = 0
    correct_rand_ratio = getRandom(min_rate, max_rate)
    incorrect_rand_ratio = getRandom(min_rate, max_rate)
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Check for our extra reinforcement goes here
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum()

        current_correct = (predicted == labels).sum()
        current_incorrect = batch_size - current_correct

        #"""
        # only issue with doing it this way is that we don't have the
        # most up to date lr for each example.
        correct_count += current_correct.item()
        incorrect_count += current_incorrect.item()
        
        if correct_count >= correct_rand_ratio:
            learning_rate = learning_rate + lr_change

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            correct_count = 0

            correct_rand_ratio = getRandom(int(min_rate), int(max_rate))
        if incorrect_count >= incorrect_rand_ratio:
            learning_rate = learning_rate - lr_change

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            incorrect_count = 0

            incorrect_rand_ratio = getRandom(int(min_rate * 2.5), int(max_rate  * 2.5))
        #"""

        optimizer.step()

        # Print stats
        running_loss += loss.item()
        if i % print_ex == 0:
            print('[%d, %5d] loss: %.3f, learning rate is %f' % (epoch, i, running_loss / 2000, learning_rate))
            running_loss = 0.0
            b_acc = (100. * train_correct.item() / train_total)
            batch_acc_output.append(b_acc)

    e_acc = (100. * train_correct.item() / train_total)
    print("Accuracy of the network for this batch: %.4f %%" % (e_acc))
    epoch_acc_output.append(e_acc)
    loss_output.append(loss.item())


end_time = time.time()
print("Finished Training in %d time" % (end_time - start_time))
print("Training acc per %d exs: " % print_ex, batch_acc_output)
print("Training acc per epoch: ", epoch_acc_output)
print("Loss per epoch:", loss_output)
print("Training time is", (end_time - start_time))
print("Final Learning rate is", learning_rate)


# Save trained model
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)

# Test the Network
correct = 0.
total = 0.
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            if batch_size == 1 and c.item():
                class_correct[label] += 1
            elif batch_size > 1:
                class_correct[label] += c[i].item()
            class_total[label] += 1
            

print('Accuracy of the network on the 10000 test images: %0.4f %%' % (100. * correct / total))

for i in range(10):
    print('Accuracy of %5s : %0.4f %%' % (
        classes[i], (100. * class_correct[i]) / class_total[i]))

batch_print = numpy.asarray(batch_acc_output)
numpy.savetxt("b_out.csv", batch_print, delimiter=",")
epoch_print = numpy.asarray(epoch_acc_output)
numpy.savetxt("e_out.csv", epoch_print, delimiter=",")
loss_print = numpy.asarray(loss_output)
numpy.savetxt("l_out.csv", loss_print, delimiter=",")



