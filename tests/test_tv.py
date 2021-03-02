from tsa.smoothing import tv
import numpy as np
import matplotlib.pyplot as plt


# basic
y = np.random.randn(100)
ys = tv.smooth(y)

fig, ax = plt.subplots()
ax.plot(y)
ax.plot(ys)


# with dependant variable
n = 100
x = np.linspace(1, 3 * np.pi, n)
y = np.random.randn(n)
ys = tv.smooth(y)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, ys)


# x masked
x = np.ma.array(np.linspace(1, 3 * np.pi, n))
x[10:25] = np.ma.masked
y = np.random.randn(n)
ys = tv.smooth(y)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, ys)

# y masked
x = np.linspace(1, 3 * np.pi, n)
y = np.ma.array(np.random.randn(n))
y[50:60] = np.ma.masked
ys = tv.smooth(y)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, ys)