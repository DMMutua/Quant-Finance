import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Import Data for S&P 500
data = yf.download("SPY")

# Computes Returns and Range
returns = np.log(data.Close / data.Close.shift(1))
range = (data.High - data.Low)
features = pd.concat([returns, range], axis=1).dropna()
features.columns = ["returns", "range"]

class GMM_Regimes:
    """
    Object to hold HMM with Gaussian Emissions.
    Assumes observations in the HMM follow a
    normal distribution.
    Params:
    n_components - Number of Hidden States. Will represent
        different market regimes
    covariance_type - parametrization level of Gaussian 
        distribution covariance matrices associated with 
        each state
    n_iter - Number of Iterations to adjust model Params by
        for maximizing likelihood of observed features given
        hidden states.
    """

    def __init__(self, n_components: int=3, covariance_type: str="full", n_iter: int = 100000):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter

    def make_model(self):
        gmm = hmm.GaussianHMM(
            n_components = self.n_components,
            covariance_type = self.covariance_type,
            n_iter = self.n_iter
        )
        return gmm


# Fit Model
Regimes = GMM_Regimes(n_iter=100000)
model = Regimes.make_model()
model.fit(features)


# Use Trained Model to Predict hidden States in Input Features
states = pd.Series(model.predict(features), index=data.index[1:])
states.name = "state"
#states.hist()
#plt.show()

# Visualizing the Regimes
color_map = {
    0.0: "green",
    1.0: "orange",
    2.0: "red"
}
(
    pd.concat([data.Close, states], axis=1)
    .dropna()
    .set_index("state", append=True)
    .Close
    .unstack("state")
    .plot(color=color_map, figsize=[16, 12])
)
plt.title('S&P 500 Regimes')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(title='Regimes')
plt.show()
