import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from data_generation import generate_double_pendulum, generate_mass_spring_damper, generate_single_pendulum
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.utils import shuffle

import random
random.seed(42069)
np.random.seed(42069)
torch.manual_seed(42069)
torch.cuda.manual_seed(42069)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ODEModel(nn.Module):

    def __init__(self, output=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100, 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, output)
        )

    def forward(self, t, y):
        return self.network(y)


def main():
    T0_train = 0
    T1_train = 1
    h_train = 0.01

    T0_test = 0
    T1_test = 1
    h_test = 0.01

    batch_size_train = 200
    batch_size_test = 200

    line_integral_model1 = ODEModel()
    line_integral_model1.to(device)

    line_integral_model2 = ODEModel()
    line_integral_model2.to(device)

    optimizer1 = Adam(line_integral_model1.parameters(), lr=1e-3)
    optimizer2 = Adam(line_integral_model2.parameters(), lr=1e-3)

    params1 = tuple(line_integral_model1.parameters())
    params2 = tuple(line_integral_model2.parameters())

    epochs = 100
    accuracies = np.zeros(epochs)

    for epoch in range(1, epochs + 1):
        y_train1, y_dot_train1 = generate_double_pendulum(batch_size_train // 2, h_train, T0_train, T1_train, diff=True, noise=True, controlled=True)
        y_train2, y_dot_train2 = generate_double_pendulum(batch_size_train // 2, h_train, T0_train, T1_train, diff=True, noise=True, controlled=False)

        optimizer1.zero_grad()

        gradients = [torch.zeros_like(param, requires_grad=False) for param in params1]

        normalization_jacobian = torch.zeros((batch_size_train // 2, y_train1.shape[2], y_train1.shape[2])).to(device)

        F = line_integral_model1(None, y_train1[0, :, :])
        F_norm = torch.linalg.norm(F, dim=1)
        for b in range(batch_size_train // 2):
            normalization_jacobian[b, :, :] = torch.diag(torch.ones(y_train1.shape[2]).to(device)) / F_norm[b] - torch.outer(F[b, :], F[b, :]) / (F_norm[b] ** 3)

        dr = y_dot_train1[0, :, :]
        dr = nn.functional.normalize(dr, dim=1)
        vjp_prev = torch.autograd.grad(F, params1, torch.bmm(normalization_jacobian, dr.unsqueeze(-1)).squeeze())

        for i in range(1, y_train1.shape[0] - 1):
            F = line_integral_model1(None, y_train1[i, :, :])
            F_norm = torch.linalg.norm(F, dim=1)
            for b in range(batch_size_train // 2):
                normalization_jacobian[b, :, :] = torch.diag(torch.ones(y_train1.shape[2]).to(device)) / F_norm[b] - torch.outer(F[b, :], F[b, :]) / (F_norm[b] ** 3)

            dr = y_dot_train1[i, :, :]
            dr = nn.functional.normalize(dr, dim=1)
            vjp_next = torch.autograd.grad(F, params1, torch.bmm(normalization_jacobian, dr.unsqueeze(-1)).squeeze())

            for j in range(len(params1)):
                gradients[j] += h_train * (vjp_next[j] + vjp_prev[j]) / 2

            vjp_prev = vjp_next

        for i, param in enumerate(line_integral_model1.parameters()):
            param.grad = - gradients[i] / batch_size_train / (T1_train - T0_train)

        optimizer1.step()

        optimizer2.zero_grad()

        gradients = [torch.zeros_like(param, requires_grad=False) for param in params2]

        normalization_jacobian = torch.zeros((batch_size_train // 2, y_train2.shape[2], y_train2.shape[2])).to(device)

        F = line_integral_model2(None, y_train2[0, :, :])
        F_norm = torch.linalg.norm(F, dim=1)
        for b in range(batch_size_train // 2):
            normalization_jacobian[b, :, :] = torch.diag(torch.ones(y_train2.shape[2]).to(device)) / F_norm[b] - torch.outer(F[b, :], F[b, :]) / (F_norm[b] ** 3)

        dr = y_dot_train2[0, :, :]
        dr = nn.functional.normalize(dr, dim=1)
        vjp_prev = torch.autograd.grad(F, params2, torch.bmm(normalization_jacobian, dr.unsqueeze(-1)).squeeze())

        for i in range(1, y_train2.shape[0] - 1):
            F = line_integral_model2(None, y_train2[i, :, :])
            F_norm = torch.linalg.norm(F, dim=1)
            for b in range(batch_size_train // 2):
                normalization_jacobian[b, :, :] = torch.diag(torch.ones(y_train2.shape[2]).to(device)) / F_norm[b] - torch.outer(F[b, :], F[b, :]) / (F_norm[b] ** 3)

            dr = y_dot_train2[i, :, :]
            dr = nn.functional.normalize(dr, dim=1)
            vjp_next = torch.autograd.grad(F, params2, torch.bmm(normalization_jacobian, dr.unsqueeze(-1)).squeeze())

            for j in range(len(params2)):
                gradients[j] += h_train * (vjp_next[j] + vjp_prev[j]) / 2

            vjp_prev = vjp_next

        for i, param in enumerate(line_integral_model2.parameters()):
            param.grad = - gradients[i] / batch_size_train / (T1_train - T0_train)

        optimizer2.step()

        with torch.no_grad():
            y_test1, y_dot_test1 = generate_double_pendulum(batch_size_test // 2, h_test, T0_test, T1_test, diff=True, noise=True, controlled=True)
            y_test2, y_dot_test2 = generate_double_pendulum(batch_size_test // 2, h_test, T0_test, T1_test, diff=True, noise=True, controlled=False)

            y_test = torch.concat([y_test1, y_test2], dim=1)
            y_dot_test = torch.concat([y_dot_test1, y_dot_test2], dim=1)

            line_integral1 = torch.zeros(batch_size_test).to(device)
            line_integral2 = torch.zeros(batch_size_test).to(device)

            F = line_integral_model1(None, y_test[0, :, :])
            F = nn.functional.normalize(F, dim=1)
            dr = y_dot_test[0, :, :]
            dr = nn.functional.normalize(dr, dim=1)
            dot_prev = torch.sum(F * dr, dim=1)

            for i in range(1, y_test.shape[0] - 1):
                F = line_integral_model1(None, y_test[i, :, :])
                F = nn.functional.normalize(F, dim=1)
                dr = y_dot_test[i, :, :]
                dr = nn.functional.normalize(dr, dim=1)
                dot_next = torch.sum(F * dr, dim=1)

                line_integral1 += h_train * (dot_next + dot_prev) / 2
                dot_prev = dot_next

            line_integral1 /= (T1_train - T0_train)

            F = line_integral_model2(None, y_test[0, :, :])
            F = nn.functional.normalize(F, dim=1)
            dr = y_dot_test[0, :, :]
            dr = nn.functional.normalize(dr, dim=1)
            dot_prev = torch.sum(F * dr, dim=1)

            for i in range(1, y_test.shape[0] - 1):
                F = line_integral_model2(None, y_test[i, :, :])
                F = nn.functional.normalize(F, dim=1)
                dr = y_dot_test[i, :, :]
                dr = nn.functional.normalize(dr, dim=1)
                dot_next = torch.sum(F * dr, dim=1)

                line_integral2 += h_train * (dot_next + dot_prev) / 2
                dot_prev = dot_next

            line_integral2 /= (T1_train - T0_train)

            L_test1 = torch.mean(line_integral1[:batch_size_test // 2])
            L_test2 = torch.mean(line_integral2[batch_size_test // 2:])

            y_test_features = torch.stack([line_integral1, line_integral2], dim=1).cpu().numpy()
            y_test_labels = np.concatenate([np.ones(batch_size_test // 2), np.zeros(batch_size_test // 2)])
            y_test_features, y_test_labels = shuffle(y_test_features, y_test_labels, random_state=42069)

            classifier = SVC()
            classifier.fit(y_test_features[:batch_size_test // 2, :], y_test_labels[:batch_size_test // 2])
            y_test_pred = classifier.predict(y_test_features[batch_size_test // 2:, :])
            accuracy = int(accuracy_score(y_test_labels[batch_size_test // 2:], y_test_pred) * 100)

        accuracies[epoch - 1] = accuracy
        print(f"Epoch: {epoch}, L_test1: {L_test1.item():.3f}, L_test2: {L_test2.item():.3f}, Test Acc: {accuracy} %")

    np.savetxt("line_integral_acc.txt", accuracies)


if __name__ == "__main__":
    main()
