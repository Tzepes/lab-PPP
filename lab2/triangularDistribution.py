import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
a = 0
c = 1
with pm.Model() as model: 
    X1 = pm.Uniform("X1", lower=0, upper=1)
    X2 = pm.Uniform("X2", lower=0, upper=1)

    Y = pm.Deterministic("Y", pm.math.abs(X1 - X2))

    Y_samples = pm.draw(Y, draws=10000)

    plt.hist(Y_samples, bins = 40)
    plt.show()