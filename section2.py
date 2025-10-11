from pygam import LinearGAM, s
import statsmodels.api as sm
import pandas as pd
import numpy as np

boston = pd.read_csv("./data/boston_housing.csv")
boston_X = boston.drop(columns=["medv"])
boston_y = boston["medv"]

b_n = boston_X.shape[0]

terms = s(0)
for i in range(1, boston_X.shape[1]):
    terms = terms + s(i)

# Build GAM with predictors
gam = LinearGAM(terms).fit(boston_X, boston_y)

X_lr = sm.add_constant(boston_X)
b_ols_model = sm.OLS(boston_y, X_lr).fit()
# Predictions OLS
y_pred_lr = b_ols_model.predict(X_lr)

lr_r2 = 0.741
lr_adjr2 = 0.734

p_lr = boston_X.shape[1] + 1

# RSS for BIC
rss_lr = ((boston_y - y_pred_lr) ** 2).sum()
bic_lr = b_n * np.log(rss_lr / b_n) + np.log(b_n) * p_lr

aic_lr = b_ols_model.aic


# Predictions GAM
y_pred_gam = gam.predict(boston_X)

gam_r2 = 0.9168

# Effective degrees of freedom
edf = gam.statistics_["edof"]
edf_total = edf if np.isscalar(edf) else sum(edf)

# Adjusted R^2
adj_r2_gam = 1 - (1 - gam_r2) * (b_n - 1) / (b_n - edf_total - 1)

# BIC approximation
rss_gam = ((boston_y - y_pred_gam) ** 2).sum()
bic_gam = b_n * np.log(rss_gam / b_n) + np.log(b_n) * edf_total

# AIC and GCV from pyGAM
aic_gam = gam.statistics_["AIC"]
gcv_gam = gam.statistics_["GCV"]

friendly_labels = {
    "lstat": "Lower Status in Neighborhood",
    "rm": "Average Number of Rooms",
    "age": "Proportion of Old Houses",
    "crim": "Per Capita Crime Rate",
    "tax": "Property Tax Rate",
    "ptratio": "Pupil-Teacher Ratio",
    "nox": "Nitric Oxide Concentration",
    "dis": "Distance to Employment Centers",
    "indus": "Proportion of Non-retail Business Acres",
    "zn": "Residential Land Zoning",
    "b": "Proportion of Black Residents",
    "chas": "Bounds Charles River (0 = No, 1 = Yes)",
    "rad": "Accessibility to Radial Highways Index",
}
