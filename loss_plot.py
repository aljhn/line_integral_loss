import numpy as np
import matplotlib.pyplot as plt


step = 2

regression = np.loadtxt("regression_loss.txt")[::step, :]
node = np.loadtxt("node_loss.txt")[::step, :]
sonode = np.loadtxt("sonode_loss.txt")[::step, :]
line_integral = np.loadtxt("line_integral_loss.txt")[::step, :]

epochs = np.arange(1, 101, step)

fig, ax = plt.subplots(1, 2)

ax[0].plot(epochs, regression[:, 0])
ax[0].plot(epochs, node[:, 0])
ax[0].plot(epochs, sonode[:, 0])
ax[0].plot(epochs, line_integral[:, 0])
ax[0].set_title("Trajectory Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].grid()

ax[0].legend(["Regression", "NODE", "SONODE", "Line Integral"])

ax[1].plot(epochs, - regression[:, 1])
ax[1].plot(epochs, - node[:, 1])
ax[1].plot(epochs, - sonode[:, 1])
ax[1].plot(epochs, - line_integral[:, 1])
ax[1].set_title("Line Integral Loss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].grid()

ax[1].legend(["Regression", "NODE", "SONODE", "Line Integral"])

fig.set_size_inches(12, 4)

# plt.show()

plt.savefig("loss.png", dpi=100)