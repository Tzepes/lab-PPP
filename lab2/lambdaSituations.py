import pymc as pm
import numpy as np
import pytensor.tensor as at
from matplotlib import pyplot as plt

# count_data = np.loadtxt("lab2/txtdata.csv")
n_count_data = len(count_data)

with pm.Model() as model:
    alpha = 1.0/20 
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_3 = pm.Exponential("lambda_3", alpha)
    tau1 = pm.DiscreteUniform("tau1", lower=0, upper=n_count_data - 1)
    tau2 = pm.DiscreteUniform("tau2", lower=0, upper=n_count_data - 1)
    tau1_S = pm.math.switch(tau2 > tau1, tau1, tau2)
    tau2_S = pm.math.switch(tau1 > tau2, tau1, tau2)

with model:
    idx = np.arange(n_count_data) # Index​
    lambda_ = pm.math.switch(tau1_S > idx, lambda_1, lambda_2)
    lambda2_ = pm.math.switch(tau2_S>idx, lambda_, lambda_3)

with model:
    data = pm.Poisson("data", lambda_)

tau, data = pm.draw([lambda2_, data])
plt.hist(data, bins = 50)
plt.hist(tau, bins = 30)
plt.show()