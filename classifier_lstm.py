import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from data_generation import generate_double_pendulum, generate_mass_spring_damper, generate_single_pendulum
from sklearn.metrics import accuracy_score

import random
random.seed(42069)
np.random.seed(42069)
torch.manual_seed(42069)
torch.cuda.manual_seed(42069)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Sigmoid(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = hn.reshape(-1, self.hidden_size)
        x = self.classifier(x)
        x = x.squeeze()
        return x


def main():
    T0_train = 0
    T1_train = 1
    h_train = 0.01

    T0_test = 0
    T1_test = 1
    h_test = 0.01

    batch_size_train = 200
    batch_size_test = 200

    classifier = LSTMClassifier(4, 300)
    classifier.to(device)

    optimizer = Adam(classifier.parameters(), lr=1e-3)

    loss_function = nn.BCELoss(reduction="mean")

    epochs = 100
    accuracies = np.zeros(epochs)

    for epoch in range(1, epochs + 1):
        x_train1 = generate_double_pendulum(batch_size_train // 2, h_train, T0_train, T1_train, diff=False, noise=True, controlled=True)
        x_train2 = generate_double_pendulum(batch_size_train // 2, h_train, T0_train, T1_train, diff=False, noise=True, controlled=False)
        x_test1 = generate_double_pendulum(batch_size_test // 2, h_test, T0_test, T1_test, diff=False, noise=False, controlled=True)
        x_test2 = generate_double_pendulum(batch_size_test // 2, h_test, T0_test, T1_test, diff=False, noise=False, controlled=False)

        x_train = torch.concat([x_train1, x_train2], dim=1)
        y_train = torch.concat([torch.ones(batch_size_train // 2), torch.zeros(batch_size_train // 2)], dim=0).to(device)

        x_test = torch.concat([x_test1, x_test2], dim=1)
        y_test = torch.concat([torch.ones(batch_size_test // 2), torch.zeros(batch_size_test // 2)], dim=0).to(device)

        optimizer.zero_grad()
        y_pred = classifier(x_train)
        L_train = loss_function(y_pred, y_train)
        L_train.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_test = classifier(x_test)
            L_test = loss_function(y_pred_test, y_test)
            y_pred_test = y_pred_test.cpu().numpy().round()
            y_test = y_test.cpu().numpy()
            accuracy = int(accuracy_score(y_test, y_pred_test) * 100)

        accuracies[epoch - 1] = accuracy
        print(f"Epoch: {epoch}, Train Loss: {L_train.item():.3f}, Test Loss: {L_test.item():.3f}, Test Acc: {accuracy} %")

    np.savetxt("lstm_acc.txt", accuracies)


if __name__ == "__main__":
    main()
