import pymc as pm
import arviz as az

def runMode():
    visitorsA = 1300
    visitorsB = 1275

    conversions_from_A = 120
    conversions_from_B = 125

    with pm.Model() as model:
        p_A = pm.Beta('p_A', alpha=1, beta=1)
        p_B = pm.Beta('p_B', alpha=1, beta=1)

        conversions_A = pm.Binomial('conversions_A', n=visitorsA, p=p_A, observed=conversions_from_A)
        conversions_B = pm.Binomial('conversions_B', n=visitorsB, p=p_B, observed=conversions_from_B)

        delta = pm.Deterministic('delta', p_B - p_A)

        trace = pm.sample(2000, tune=1000, return_inferencedata=True)

    print(az.summary(trace, hdi_prob=0.95))
    az.plot_posterior(trace, var_names=['p_A', 'p_B', 'delta'], hdi_prob=0.95)

if __name__ == "__main__":
    runMode()