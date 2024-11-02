import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

c = 10

# Create model context
with pm.Model() as model:
    # Define a uniform distribution with a name
    X = pm.Uniform("X", lower=0, upper=4 * c)

    # Define the deterministic function l
    # l = pm.Deterministic('l', pm.math.minimum(c, X) / (c + X))
    l = pm.Deterministic('l', X/ (c + X))

    # Output the graph of the model
    pm.model_to_graphviz(model).render("model_graph", format="png")

    print(l.eval())