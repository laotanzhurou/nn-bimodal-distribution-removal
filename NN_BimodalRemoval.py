# import libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

# Hyper Parameters
input_size = 4
hidden_size = 10
num_classes = 2
num_epochs = 500
learning_rate = 0.02

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


def bimodal_remove(X, Y, output, sigma):
    p_error = abs(Y.numpy() - output.detach().numpy())
    pe_mean = p_error.mean()
    pe_std = p_error.std()

    # plot error distribution
    n, bins, patches = plt.hist(p_error, 10, range=[0, 2], facecolor='blue')
    # plt.xlabel('Error')
    # plt.ylabel('Frequency')
    # plt.title(r'Histogram of IQ: $\mu='+str(pe_mean)+'$, $\sigma='+str(pe_std)+'$')
    # plt.tight_layout()
    # plt.show()

    # create candidate array
    candidate = []
    for i in range(0, len(p_error)):
        if p_error[i] > pe_mean:
            candidate.append(p_error[i])
    candidate = np.array(candidate)

    # create removal array
    c_mean = candidate.mean()
    c_std = candidate.std()
    removal = []
    for j in range(0, len(candidate)):
        if candidate[j] > c_mean + sigma * c_std:
            removal.append(candidate[j])
    removal = np.array(removal)

    # prepare index to remove
    removal_index = []
    for k in range(0, len(p_error)):
        for l in range(0, len(removal)):
            if p_error[k] == removal[l]:
                removal_index.append(k)

    # remove from training set
    x_num = X.numpy()
    y_num = Y.numpy()
    x_num = np.delete(x_num, removal_index, 0)
    y_num = np.delete(y_num, removal_index, 0)
    updated_x = torch.from_numpy(x_num)
    updated_y = torch.from_numpy(y_num)
    return updated_x, updated_y


def training(X, Y, sigma, variance_threshold):
    net = Net(input_size, hidden_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # store all losses for visualisation
    all_losses = []

    # train the model
    end_epoch = 0
    for epoch in range(num_epochs):
        end_epoch = epoch
        # Forward
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(X)

        # Compute loss
        loss = criterion(outputs, Y)
        all_losses.append(loss.item())

        # Terminal if variance is lower than threshold
        if (loss <= variance_threshold):
            # print("Loss reduced to " + str(variance_threshold) + " terminating process at Epoch " + str(epoch))
            break

        # Bimodal Removal
        if (epoch % 50 == 0):
            _, predicted = torch.max(outputs, 1)
            # calculate and print accuracy
            total = predicted.size(0)
            correct = predicted.data.numpy() == Y.data.numpy()
            # print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
            #       % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct) / total))
            # perform bimodal removal
            X, Y = bimodal_remove(X, Y, _, sigma)
            # print("Training Set Size after Bimodal Removal: " + str(Y.size()[0]))

        # Backward
        net.zero_grad()
        loss.backward()
        optimizer.step()

    # Optional: plotting historical loss from ``all_losses`` during network learning
    # Please uncomment me from next line to ``plt.show()`` if you want to plot loss

    #
    # plt.figure()
    # plt.plot(all_losses)
    # plt.show()

    return net, end_epoch + 1, Y.size()[0]


def evaluate(net, train_data, test_data):
    train_input = train_data.iloc[:, :input_size]
    train_target = train_data.iloc[:, input_size]

    inputs = torch.Tensor(train_input.values).float()
    training_targets = torch.Tensor(train_target.values).long()

    outputs = net(inputs)
    _, training_predicted = torch.max(outputs, 1)

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
    return sum(correct) / total


test_sigma = 2
test_variance_threshold = 0.01
rounds = 10
all_accuracy = []
all_bdr_size = []
all_ending_epoch = []

for a in range(0, rounds):
    # load all data
    train_data = pd.read_csv('fzis_training.csv', header=None)
    test_data = pd.read_csv('fzis_testing.csv', header=None)

    # define train dataset and a data loader
    train_input = train_data.iloc[:, :input_size]
    train_target = train_data.iloc[:, input_size]

    X = torch.Tensor(train_input.values).float()
    Y = torch.Tensor(train_target.values).long()

    net, result_epoch, bdr_size = training(X, Y, test_sigma, test_variance_threshold)
    accuracy = evaluate(net, train_data, test_data)

    all_accuracy.append(accuracy)
    all_bdr_size.append(bdr_size)
    all_ending_epoch.append(result_epoch)

mean_accuracy = np.array(all_accuracy).mean()
mean_bdr_size = np.array(all_bdr_size).mean()
mean_end_epoch = np.array(all_ending_epoch).mean()

print("Result from sigma: " + str(test_sigma) + " variance_threshold: " + str(test_variance_threshold)
      + " average accuracy: " + str(mean_accuracy)
      + " ending epoch: " + str(mean_end_epoch)
      + " remaining input set: " + str(mean_bdr_size))

