import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torchdiffeq.torchdiffeq import odeint_adjoint as odeint
from data_generation import generate_double_pendulum, generate_mass_spring_damper, generate_single_pendulum

import random
random.seed(42069)
np.random.seed(42069)
torch.manual_seed(42069)
torch.cuda.manual_seed(42069)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ODEModel(nn.Module):

    def __init__(self):
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
            nn.Linear(50, 2)
        )

    def forward(self, t, y):
        y_dot = y[:, 2:]
        y_ddot = self.network(y)
        return torch.concat([y_dot, y_ddot], dim=1)


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

    node_model = ODEModel()
    node_model.to(device)

    optimizer = Adam(node_model.parameters(), lr=1e-3)

    node_loss_function = nn.MSELoss(reduction="mean")
    test_loss_function = nn.MSELoss(reduction="mean")

    epochs = 100
    losses = np.zeros((epochs, 2))

    for epoch in range(1, epochs + 1):
        y_train, y_dot_train = generate_double_pendulum(batch_size_train, h_train, T0_train, T1_train, diff=True, noise=True)
        y_test, y_dot_test = generate_double_pendulum(batch_size_test, h_test, T0_test, T1_test, diff=True, noise=False)

        optimizer.zero_grad()
        y0_train = y_train[0, :, :]
        y_train_pred = odeint(node_model, y0_train, t_train, method="rk4")
        L_train = node_loss_function(y_train_pred, y_train)
        L_train.backward()
        optimizer.step()

        with torch.no_grad():
            y0_test = y_test[0, :, :]
            y_test_pred = odeint(node_model, y0_test, t_test, method="rk4")
            L_test = test_loss_function(y_test_pred, y_test)

            line_integral = torch.zeros(batch_size_test).to(device)

            F = node_model(None, y_test[0, :, :])
            F = nn.functional.normalize(F, dim=1)
            dr = y_dot_test[0, :, :]
            dr = nn.functional.normalize(dr, dim=1)
            dot_prev = torch.sum(F * dr, dim=1)

            for i in range(1, y_test.shape[0] - 1):
                F = node_model(None, y_test[i, :, :])
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
        print(f"Epoch: {epoch}, Train: {L_train.item():.3f}, Test: {L_test.item():.3f}, Test Int: {test_int.item():.3f}")

    np.savetxt("sonode_loss.txt", losses)

    exit()

    node_model.to(torch.device("cpu"))

    x = np.arange(-5, 5, 0.1)
    n = len(x)

    X = torch.zeros((n * n, 2))
    for i in range(n):
        for j in range(n):
            X[i + n * j, 0] = x[j]
            X[i + n * j, 1] = x[i]

    with torch.no_grad():
        Y = node_model(None, X)

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
