# import libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np


# Hyper Parameters
input_size = 4
num_classes = 2
num_epochs = 500


# define a function to plot confusion matrix
def plot_confusion(input_sample, num_classes, des_output, actual_output):
    confusion = torch.zeros(num_classes, num_classes)
    for i in range(input_sample):
        actual_class = actual_output[i]
        predicted_class = des_output[i]

        confusion[actual_class][predicted_class] += 1

    return confusion

"""
Step 1: Load data and pre-process data
Here we use data loader to read data
"""


# load all data
train_data = pd.read_csv('fzis_training.csv', header=None)
test_data = pd.read_csv('fzis_testing.csv', header=None)

# Neural Network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out


def training(hidden_size, learning_rate):
    # define train dataset and a data loader
    train_input = train_data.iloc[:, :input_size]
    train_target = train_data.iloc[:, input_size]

    X = torch.Tensor(train_input.values).float()
    Y = torch.Tensor(train_target.values).long()

    net = Net(input_size, hidden_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # store all losses for visualisation
    all_losses = []

    # train the model
    for epoch in range(num_epochs):

        # Forward
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(X)

        # Compute loss
        loss = criterion(outputs, Y)
        all_losses.append(loss.item())

        if (epoch % 50 == 0 or epoch == num_epochs - 1):
            output, predicted = torch.max(outputs, 1)
            # calculate and print accuracy
            total = predicted.size(0)
            correct = predicted.data.numpy() == Y.data.numpy()

            # print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
            #       % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct) / total))

            # plot pattern errors
            p_error = abs(Y.numpy() - output.detach().numpy())
            pe_mean = p_error.mean()
            pe_std = p_error.std()

            # plot error distribution
            # n, bins, patches = plt.hist(p_error, 10, range=[0, 2], facecolor='blue')
            # plt.xlabel('Error')
            # plt.ylabel('Frequency')
            # plt.title(r'Histogram of IQ: $\mu=' + str(pe_mean) + '$, $\sigma=' + str(pe_std) + '$')
            # plt.tight_layout()
            # plt.show()

        # Backward
        net.zero_grad()
        loss.backward()
        optimizer.step()

    # Optional: plotting historical loss from ``all_losses`` during network learning
    # Please uncomment me from next line to ``plt.show()`` if you want to plot loss

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.plot(all_losses)
    # plt.show()

    """
    Evaluating the Results

    To see how well the network performs on different categories, we will
    create a confusion matrix, indicating for every glass (rows)
    which class the network guesses (columns).

    """

    train_input = train_data.iloc[:, :input_size]
    train_target = train_data.iloc[:, input_size]

    inputs = torch.Tensor(train_input.values).float()
    targets = torch.Tensor(train_target.values).long()

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    # print('Confusion matrix for training:')
    # print(plot_confusion(train_input.shape[0], num_classes, predicted.long().data, targets.data))

    """
    Step 3: Test the neural network

    Pass testing data to the built neural network and get its performance
    """
    # get testing data
    test_input = test_data.iloc[:, :input_size]
    test_target = test_data.iloc[:, input_size]

    inputs = torch.Tensor(test_input.values).float()
    targets = torch.Tensor(test_target.values).long()

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    total = predicted.size(0)
    correct = predicted.data.numpy() == targets.data.numpy()

    # print('Testing Accuracy: %.2f %%' % (100 * sum(correct) / total))

    """
    Evaluating the Results

    To see how well the network performs on different categories, we will
    create a confusion matrix, indicating for every glass (rows)
    which class the network guesses (columns).

    """

    # print('Confusion matrix for testing:')
    # print(plot_confusion(test_input.shape[0], num_classes, predicted.long().data, targets.data))

    # return accuracy rate
    return sum(correct) / total


# perform training
hidden_unit = 10
learning_rate = 0.02
rounds = 10

all_accuracy = []
for a in range(0, rounds):
    accuracy = training(hidden_unit, learning_rate)
    all_accuracy.append(accuracy)

mean_accuracy = np.array(all_accuracy).mean()
print("Result from hidden unit: " + str(hidden_unit) + " lr: " + str(learning_rate) + " average accuracy: " + str(mean_accuracy))
