import matplotlib.pyplot as plt


# polynomial detrend example
N, k = 100, 10
q, m, c = np.random.randn(3, k, 1)
x = np.arange(N)
Y = q*x*x + m * x + c * N

Ydt = poly_uniform_bulk(Y, 2)

fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
ax1.plot(Y.T)
ax2.plot(Ydt.T)
