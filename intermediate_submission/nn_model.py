import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

class ClassificationNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 152)
        self.fc2 = nn.Linear(152, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
def train_loop(dataloader, optimizer, criterion, clf, epochs=10):
    clf.train()

    for epoch in range(epochs):

        running_loss = 0.0
        for inputs, labels in dataloader:
            # get the inputs (data is in a list of [X_train, y_train])
            inputs = inputs.float()
            labels = labels[:, None].float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = clf(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print verbose
            running_loss += loss.item()
        
        print(f"Epoch: {epoch}, final loss: {running_loss:.3f}")

    print('Finish training')

def eval_loop(dataloader, clf):
    clf.eval()
    preds = []
    probas = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            x_test, y_test = data
            x_test = x_test.float()

            output = clf(x_test)
            y_test = torch.flatten(y_test).tolist()
            proba = torch.flatten(output).tolist()
            pred = list(map(lambda x: 1 if x >= 0.5 else 0, proba))
            preds += pred
            probas += proba
            labels += y_test
    
    return preds, probas, labels