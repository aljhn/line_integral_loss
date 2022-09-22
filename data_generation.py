import numpy as np
import torch
import torch.nn as nn
from torchdiffeq.torchdiffeq import odeint_adjoint as odeint

import random
random.seed(42069)
np.random.seed(42069)
torch.manual_seed(42069)
torch.cuda.manual_seed(42069)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DoublePendulumSystem(nn.Module):

    def __init__(self, controlled=False):
        super().__init__()
        self.m1 = 1
        self.m2 = 1
        self.l1 = 1
        self.l2 = 1
        self.g = 9.81

        self.controlled = controlled

    def forward(self, t, y):
        batch_size = y.shape[0]

        M = torch.zeros((batch_size, 2, 2))
        M[:, 0, 0] = (self.m1 + self.m2) * (self.l1**2)
        M[:, 0, 1] = self.m2 * self.l1 * self.l2 * torch.cos(y[:, 0] - y[:, 1])
        M[:, 1, 0] = self.m2 * self.l1 * self.l2 * torch.cos(y[:, 0] - y[:, 1])
        M[:, 1, 1] = self.m2 * (self.l2**2)

        M_det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]

        M_inv = torch.zeros((batch_size, 2, 2))
        M_inv[:, 0, 0] = M[:, 1, 1] / M_det
        M_inv[:, 0, 1] = - M[:, 0, 1] / M_det
        M_inv[:, 1, 0] = - M[:, 1, 0] / M_det
        M_inv[:, 1, 1] = M[:, 0, 0] / M_det

        ff = torch.zeros((batch_size, 2, 1))
        ff[:, 0, 0] = - self.m2 * self.l1 * self.l2 * torch.sin(y[:, 0] - y[:, 1]) * (y[:, 3]**2) - (self.m1 + self.m2) * self.g * self.l1 * torch.sin(y[:, 0])
        ff[:, 1, 0] = self.m2 * self.l1 * self.l2 * torch.sin(y[:, 0] - y[:, 1]) * (y[:, 2]**2) - self.m2 * self.g * self.l2 * torch.sin(y[:, 1])

        if self.controlled:
            u1 = -0.1 * y[:, 0] - 0.01 * y[:, 2]
            u2 = -5.0 * (y[:, 1] - y[:, 0]) - 2.0 * y[:, 3]
            u = torch.stack([u1, u2], dim=1).unsqueeze(2)
            ff += u

        yy = torch.matmul(M_inv, ff)

        return torch.stack([y[:, 2], y[:, 3], yy[:, 0, 0], yy[:, 1, 0]], dim=1)


class MassSpringDamperSystem(nn.Module):

    def __init__(self):
        super().__init__()
        self.m = 1
        self.d = 1
        self.k = 1
        self.A = torch.tensor([[0, 1], [-self.k / self.m, -self.d / self.m]])

    def forward(self, t, y):
        return torch.matmul(y, self.A.T)


class SinglePendulumSystem(nn.Module):

    def __init__(self):
        super().__init__()
        self.l = 1
        self.g = 9.81

    def forward(self, t, y):
        y_dot = y[:, 1:2]
        y_ddot = - self.g / self.l * torch.sin(y[:, 0:1])
        return torch.concat([y_dot, y_ddot], dim=1)


def generate_double_pendulum(n, h, T0, T1, diff=True, noise=True, controlled=False):
    y0 = torch.zeros((n, 4))
    y0[:, 0] = torch.rand(n) * 6 - 3
    y0[:, 1] = y0[:, 0] + torch.rand(n) - 0.5
    y0[:, 2] = torch.rand(n) * 4 - 2
    y0[:, 3] = torch.rand(n) * 4 - 2

    t = torch.arange(T0, T1, h)
    with torch.no_grad():
        y = odeint(DoublePendulumSystem(controlled), y0, t, method="rk4")

    if noise:
        y += torch.randn_like(y) * 0.01

    if not diff:
        return y.to(device)

    y_dot = (y[1:, :, :] - y[:-1, :, :]) / h

    return y.to(device), y_dot.to(device)


def generate_mass_spring_damper(n, h, T0, T1, diff=True, noise=True):
    y0 = torch.rand((n, 2)) * 6 - 3

    t = torch.arange(T0, T1, h)
    with torch.no_grad():
        y = odeint(MassSpringDamperSystem(), y0, t, method="rk4")

    if noise:
        y += torch.randn_like(y) * 0.01

    if not diff:
        return y.to(device)

    y_dot = (y[1:, :, :] - y[:-1, :, :]) / h

    return y.to(device), y_dot.to(device)


def generate_single_pendulum(n, h, T0, T1, diff=True, noise=True):
    y0 = torch.rand((n, 2)) * 12 - 6

    t = torch.arange(T0, T1, h)
    with torch.no_grad():
        y = odeint(SinglePendulumSystem(), y0, t, method="rk4")

    if noise:
        y += torch.randn_like(y) * 0.01

    if not diff:
        return y.to(device)

    y_dot = (y[1:, :, :] - y[:-1, :, :]) / h

    return y.to(device), y_dot.to(device)


def simulate_double_pendulum():
    y0 = torch.zeros((1, 4))
    y0[:, 0] = torch.rand(1) * 6 - 3
    y0[:, 1] = y0[:, 0] + torch.rand(1) - 0.5
    y0[:, 2] = torch.rand(1) * 4 - 2
    y0[:, 3] = torch.rand(1) * 4 - 2

    T1 = 10
    t = torch.arange(0, T1, 0.01)

    y = odeint(DoublePendulumSystem(True), y0, t, method="rk4")
    theta_1 = y[:, 0, 0]
    theta_2 = y[:, 0, 1]
    theta_1_dot = y[:, 0, 2]
    theta_2_dot = y[:, 0, 3]

    with open("double_pendulum.js", "w") as f:

        f.write("var theta_1 = [")
        for i in range(theta_1.shape[0]):
            f.write(str(theta_1[i].item()))
            if i < theta_1.shape[0] - 1:
                f.write(", ")
        f.write("];\n\n")

        f.write("var theta_2 = [")
        for i in range(theta_2.shape[0]):
            f.write(str(theta_2[i].item()))
            if i < theta_2.shape[0] - 1:
                f.write(", ")
        f.write("];\n\n")

        f.write("var theta_1_dot = [")
        for i in range(theta_1_dot.shape[0]):
            f.write(str(theta_1_dot[i].item()))
            if i < theta_1_dot.shape[0] - 1:
                f.write(", ")
        f.write("];\n\n")

        f.write("var theta_2_dot = [")
        for i in range(theta_2_dot.shape[0]):
            f.write(str(theta_2_dot[i].item()))
            if i < theta_2_dot.shape[0] - 1:
                f.write(", ")
        f.write("];\n\n")


if __name__ == "__main__":
    simulate_double_pendulum()
