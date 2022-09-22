import numpy as np
import matplotlib.pyplot as plt


step = 1

lstm = np.loadtxt("lstm_acc.txt")[::step]
line_integral = np.loadtxt("line_integral_acc.txt")[::step]

epochs = np.arange(1, 101, step)

fig, ax = plt.subplots(1, 1)

ax.plot(epochs, lstm)
ax.plot(epochs, line_integral)
ax.set_title("Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("%")
ax.grid()

ax.legend(["LSTM", "Line Integral + SVC"])

fig.set_size_inches(12, 4)

# plt.show()

plt.savefig("acc.png", dpi=100)