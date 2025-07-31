import numpy as np
import matplotlib.pyplot as plt

a = np.array([0, 0])
b = np.array([4, 4])

for i in range(60):
    t = i / 59
    p = (1 - t)*a + t*b
    plt.clf() #clear current figure
    plt.plot([a[0], b[0]], [a[1], b[1]], 'k--', alpha=0.5) #0.5 for semi transparent k-- for black
    plt.plot(p[0], p[1], 'ro')
    plt.xlim(-1, 5) #set x limits so the plot does't autot rescale
    plt.ylim(-1, 5)
    plt.pause(0.05)  # simulate 20 FPS