import time
import  torch
from torch import Tensor
import numpy as np


def trainet(data_loader, net , num_epochs, criterion, optimizer):
    train_start_time = time.time()
    num_batches = len(data_loader)
    acc = []
    losses = []
    val_loss_list = []
    trn_loss=[]
    avg_train_loss=[]
    acc_list = []
    device = 'cuda'
    for epoch in range(num_epochs):

        start_time = time.time()
        for i, (inputs, targets) in enumerate(data_loader, 0):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
            # grad init
            optimizer.zero_grad()
            # forward propagation
            model_output = net(inputs)
            # calculate loss
            targets = targets.squeeze_()
            loss = criterion(model_output, targets)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()
            trn_loss.append(loss.item())

            # Track the accuracy
            total = targets.size(0)
            _, predicted = torch.max(model_output.data, 1)
            correct = (predicted == targets).sum().item()
            #acc_list.append(correct / total)

            # print(epoch,i,trn_loss)
            # print(model_output)

            if (i + 1) % 20 == 0:
               print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, num_batches, loss.item(), (correct / total) * 100))
        train_loss=np.average(trn_loss)
        avg_train_loss.append(train_loss)
        trn_losses = []

    return net, avg_train_loss


def prediction(test_loader,net):
    net.eval()
    device = 'cuda'
    all_preds= torch.tensor([])
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
            all_preds=all_preds.to(device)
            outputs = net(inputs)

            targets = targets.squeeze_()
            _, predicted = torch.max(outputs.data, 1)
            p=predicted.type(Tensor)
            all_preds = torch.cat((all_preds, p.to(device)), dim=0)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
    return  (correct / total)