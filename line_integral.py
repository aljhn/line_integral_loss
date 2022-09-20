import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torchdiffeq.torchdiffeq import odeint_adjoint as odeint
from data_generation import generate_double_pendulum, generate_mass_spring_damper, generate_single_pendulum, DoublePendulumSystem

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


class CombinedModel(nn.Module):

    def __init__(self, line_integral_model, regression_model):
        super().__init__()
        self.line_integral_model = line_integral_model
        self.regression_model = regression_model

    def forward(self, t, y):
        direction = nn.functional.normalize(self.line_integral_model(t, y), dim=1)
        magnitude = self.regression_model(t, y)
        return direction * magnitude


def main():
    T0_train = 0
    T1_train = 1
    h_train = 0.01
    t_train = torch.arange(T0_train, T1_train, h_train).to(device)

    T0_test = 0
    T1_test = 1
    h_test = 0.01
    t_test = torch.arange(T0_test, T1_test, h_test).to(device)

    batch_size_train = 200
    batch_size_test = 20

    line_integral_model = ODEModel()
    line_integral_model.to(device)

    regression_model = ODEModel(1)
    regression_model.to(device)

    combined_model = CombinedModel(line_integral_model, regression_model)
    combined_model.to(device)

    optimizer_line_integral = Adam(line_integral_model.parameters(), lr=1e-3)
    optimizer_regression = Adam(regression_model.parameters(), lr=1e-3)

    params = tuple(line_integral_model.parameters())

    regression_loss_function = nn.MSELoss(reduction="mean")
    test_loss_function = nn.MSELoss(reduction="mean")

    epochs = 100
    losses = np.zeros((epochs, 2))

    for epoch in range(1, epochs + 1):
        y_train, y_dot_train = generate_double_pendulum(batch_size_train, h_train, T0_train, T1_train, diff=True, noise=True)
        y_test, y_dot_test = generate_double_pendulum(batch_size_test, h_test, T0_test, T1_test, diff=True, noise=False)

        y_train_regression = torch.flatten(y_train[:-1, :, :], start_dim=0, end_dim=1)
        y_dot_train_regression = torch.linalg.norm(torch.flatten(y_dot_train, start_dim=0, end_dim=1), dim=1).unsqueeze(1)

        optimizer_line_integral.zero_grad()

        gradients = [torch.zeros_like(param, requires_grad=False) for param in params]

        normalization_jacobian = torch.zeros((batch_size_train, y_train.shape[2], y_train.shape[2])).to(device)

        F = line_integral_model(None, y_train[0, :, :])
        F_norm = torch.linalg.norm(F, dim=1)
        for b in range(batch_size_train):
            normalization_jacobian[b, :, :] = torch.diag(torch.ones(y_train.shape[2]).to(device)) / F_norm[b] - torch.outer(F[b, :], F[b, :]) / (F_norm[b] ** 3)

        dr = y_dot_train[0, :, :]
        dr = nn.functional.normalize(dr, dim=1)
        vjp_prev = torch.autograd.grad(F, params, torch.bmm(normalization_jacobian, dr.unsqueeze(-1)).squeeze())

        for i in range(1, y_train.shape[0] - 1):
            F = line_integral_model(None, y_train[i, :, :])
            F_norm = torch.linalg.norm(F, dim=1)
            for b in range(batch_size_train):
                normalization_jacobian[b, :, :] = torch.diag(torch.ones(y_train.shape[2]).to(device)) / F_norm[b] - torch.outer(F[b, :], F[b, :]) / (F_norm[b] ** 3)

            dr = y_dot_train[i, :, :]
            dr = nn.functional.normalize(dr, dim=1)
            vjp_next = torch.autograd.grad(F, params, torch.bmm(normalization_jacobian, dr.unsqueeze(-1)).squeeze())

            for j in range(len(params)):
                gradients[j] += h_train * (vjp_next[j] + vjp_prev[j]) / 2

            vjp_prev = vjp_next

        for i, param in enumerate(line_integral_model.parameters()):
            param.grad = - gradients[i] / batch_size_train / (T1_train - T0_train)

        optimizer_line_integral.step()

        with torch.no_grad():
            line_integral = torch.zeros(batch_size_train).to(device)

            F = line_integral_model(None, y_train[0, :, :])
            F = nn.functional.normalize(F, dim=1)
            dr = y_dot_train[0, :, :]
            dr = nn.functional.normalize(dr, dim=1)
            dot_prev = torch.sum(F * dr, dim=1)

            for i in range(1, y_train.shape[0] - 1):
                F = line_integral_model(None, y_train[i, :, :])
                F = nn.functional.normalize(F, dim=1)
                dr = y_dot_train[i, :, :]
                dr = nn.functional.normalize(dr, dim=1)
                dot_next = torch.sum(F * dr, dim=1)

                line_integral += h_train * (dot_next + dot_prev) / 2
                dot_prev = dot_next

            line_integral /= (T1_train - T0_train)
            L_train = torch.mean(line_integral)

        optimizer_regression.zero_grad()
        y_dot_pred_regression = regression_model(None, y_train_regression)
        L_train_regression = regression_loss_function(y_dot_pred_regression, y_dot_train_regression)
        L_train_regression.backward()
        optimizer_regression.step()

        with torch.no_grad():
            y0_test = y_test[0, :, :]
            y_test_pred = odeint(combined_model, y0_test, t_test, method="rk4")
            L_test = test_loss_function(y_test_pred, y_test)

            line_integral = torch.zeros(batch_size_test).to(device)

            F = line_integral_model(None, y_test[0, :, :])
            F = nn.functional.normalize(F, dim=1)
            dr = y_dot_test[0, :, :]
            dr = nn.functional.normalize(dr, dim=1)
            dot_prev = torch.sum(F * dr, dim=1)

            for i in range(1, y_test.shape[0] - 1):
                F = line_integral_model(None, y_test[i, :, :])
                F = nn.functional.normalize(F, dim=1)
                dr = y_dot_test[i, :, :]
                dr = nn.functional.normalize(dr, dim=1)
                dot_next = torch.sum(F * dr, dim=1)

                line_integral += h_test * (dot_next + dot_prev) / 2
                dot_prev = dot_next

            line_integral /= (T1_test - T0_test)
            test_int = torch.mean(line_integral)

        losses[epoch - 1, 0] = L_test.item()
        losses[epoch - 1, 1] = test_int.item()
        print(f"Epoch: {epoch}, Int: {L_train.item():.3f}, Reg: {L_train_regression.item():.3f}, Test: {L_test.item():.3f}, Test Int: {test_int.item():.3f}")

    np.savetxt("line_integral_loss.txt", losses)

    exit()

    line_integral_model.to(torch.device("cpu"))
    regression_model.to(torch.device("cpu"))

    x = np.arange(-5, 5, 0.1)
    n = len(x)

    X = torch.zeros((n * n, 2))
    for i in range(n):
        for j in range(n):
            X[i + n * j, 0] = x[j]
            X[i + n * j, 1] = x[i]

    with torch.no_grad():
        Y = line_integral_model(None, X) * regression_model(None, X)

    X1 = np.zeros((n, n))
    X2 = np.zeros((n, n))
    Y1 = np.zeros((n, n))
    Y2 = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            X1[i, j] = X[i + n * j, 0]
            X2[i, j] = X[i + n * j, 1]
            Y1[i, j] = Y[i + n * j, 0]
            Y2[i, j] = Y[i + n * j, 1]

    plt.figure()
    plt.streamplot(X1, X2, Y1, Y2, density=1.0, linewidth=None, color="#A23BEC")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.show()


if __name__ == "__main__":
    main()
