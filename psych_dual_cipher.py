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

batch_size = 10

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
correct_learning_rate = 0.025
cor_lr_change = correct_learning_rate * 0.0004
incorrect_learning_rate = 0.025
incor_lr_change = incorrect_learning_rate * 0.00225

momentum = 0.9
num_epochs = 10

epoch_acc_output = []
epoch_test_acc_output = []
cor_lr_output = []
incor_lr_output = []
min_rate = 745
max_rate = 755

# Loss Function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=correct_learning_rate, momentum=momentum)
optimizer = optim.Adagrad(model.parameters(), lr=correct_learning_rate)
#optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
#optimizer = optim.Adam(model.parameters(), lr=correct_learning_rate)

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

        # Check for our extra reinforcement
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum()

        current_correct = (predicted == labels).sum()
        current_incorrect = batch_size - current_correct
        
        # only issue with doing it this way within batches is that
        # we don't have the most up to date lr for each example.
        # However, the performance increase with batches makes it worthwhile

        # Update learning rate based on correct and incorrect responses
        ex_correct_or_incorrect = 0
        if current_correct >= current_incorrect:
            correct_count += current_correct
            incorrect_count += current_incorrect
            ex_correct_or_incorrect = 1
        else:
            correct_count += current_correct
            incorrect_count += current_incorrect
            ex_correct_or_incorrect = 0

        if correct_count >= correct_rand_ratio:
            correct_learning_rate = correct_learning_rate - cor_lr_change
            correct_count = 0
            correct_rand_ratio = getRandom(int(min_rate), int(max_rate))
        if incorrect_count >= incorrect_rand_ratio:
            incorrect_learning_rate = incorrect_learning_rate + incor_lr_change
            incorrect_count = 0
            incorrect_rand_ratio = getRandom(int(min_rate * 2.5), int(max_rate  * 2.5))


        for param_group in optimizer.param_groups:
            if ex_correct_or_incorrect:
                param_group['lr'] = correct_learning_rate
            else:
                param_group['lr'] = incorrect_learning_rate
        
        optimizer.step()

        # Print stats
        running_loss += loss.item()
        
    # Test the Network
    test_correct = 0.
    test_total = 0.
    with torch.no_grad():
        for data in testloader:
            test_images, test_labels = data
            test_outputs = model(test_images)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()

            c_test = (test_predicted == test_labels).squeeze()
            if (len(c_test.size()) == 0):
                continue
            
    test_acc = (100. * test_correct / test_total)
    epoch_test_acc_output.append(test_acc)
    cor_lr_output.append(correct_learning_rate)
    incor_lr_output.append(incorrect_learning_rate)
    e_acc = (100. * train_correct.item() / train_total)
    print("Accuracy of the network for this batch: %.4f %%" % (e_acc))
    epoch_acc_output.append(e_acc)

    

end_time = time.time()
print("Finished Training in %d time" % (end_time - start_time))
print("Training acc per epoch: ", epoch_acc_output)
print("Training acc per epoch: ", epoch_test_acc_output)
print("Training time is", (end_time - start_time))
print("Final Correct Learning rate is", correct_learning_rate)
print("Final Incorrect Learning rate is", incorrect_learning_rate)



# Save trained model
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)


epoch_print = numpy.asarray(epoch_acc_output)
numpy.savetxt("e_out.csv", epoch_print, delimiter=",")
incor_lr_print = numpy.asarray(incor_lr_output)
numpy.savetxt("incor_lr_out.csv", incor_lr_print, delimiter=",")
cor_lr_print = numpy.asarray(cor_lr_output)
numpy.savetxt("cor_lr_out.csv", cor_lr_print, delimiter=",")
test_epoch_print = numpy.asarray(epoch_test_acc_output)
numpy.savetxt("t_out.csv", test_epoch_print, delimiter=",")



