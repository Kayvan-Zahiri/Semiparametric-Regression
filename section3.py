import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import scipy.sparse as sp
from statsmodels.gam.api import GLMGam, BSplines
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.anova import anova_lm
from pygam import LinearGAM, s

happiness = pd.read_csv("./data/happiness/2019.csv")

rename_map = {
    "GDP per capita": "gdp",
    "Social support": "support",
    "Healthy life expectancy": "lifeexp",
    "Freedom to make life choices": "freedom",
    "Generosity": "generosity",
    "Perceptions of corruption": "corruption",
    "Score": "score",
    "Country or region": "country",
    "Overall rank": "rank",
    "Score": "score",
}
happiness = happiness.rename(columns=rename_map)
y_col = "score"
X_cols = list(happiness.columns)
X_cols.remove("rank")
X_cols.remove("score")
X_cols.remove("country")

# Restore the old .A attribute (alias for .toarray())
if not hasattr(sp.csr_matrix, "A"):
    sp.csr_matrix.A = property(lambda self: self.toarray())

x_options = ["gdp", "support", "lifeexp", "freedom", "generosity", "corruption"]


# --- Helper functions ---
def fitted_linear(col, data):
    X = sm.add_constant(data[col])
    y = data["score"]
    model = sm.OLS(y, X).fit()
    x_sorted = np.linspace(data[col].min(), data[col].max(), 100)
    X_pred = sm.add_constant(x_sorted)
    y_pred = model.predict(X_pred)
    return x_sorted, y_pred


def fitted_gam(col, data):
    X = np.asarray(data[[col]].to_numpy(), dtype=float)
    y = np.asarray(data["score"].to_numpy(), dtype=float)
    gam = LinearGAM(s(0, n_splines=6)).fit(X, y)
    x_sorted = np.linspace(X.min(), X.max(), 100)[:, None]
    y_pred = gam.predict(x_sorted)
    return x_sorted.flatten(), y_pred


def make_traces(col, data):
    # linear fit
    x_lin, y_lin = fitted_linear(col, data)
    trace_lin = go.Scatter(
        x=x_lin, y=y_lin, mode="lines", line=dict(color="red"), name="Linear"
    )

    # gam fit
    x_gam, y_gam = fitted_gam(col, data)
    trace_gam = go.Scatter(
        x=x_gam, y=y_gam, mode="lines", line=dict(color="blue"), name="GAM"
    )

    # scatter points
    scatter_lin = go.Scatter(
        x=data[col],
        y=data["score"],
        mode="markers",
        marker=dict(size=6, color="gray", opacity=0.6),
        showlegend=False,
    )
    scatter_gam = go.Scatter(
        x=data[col],
        y=data["score"],
        mode="markers",
        marker=dict(size=6, color="gray", opacity=0.6),
        showlegend=False,
    )
    return [trace_lin, trace_gam, scatter_lin, scatter_gam]


def residuals_linear(col, data):
    X = sm.add_constant(data[col])
    y = data["score"]
    model = sm.OLS(y, X).fit()
    fitted = model.predict(X)
    return y - fitted


def residuals_gam(col, data):
    X = np.asarray(data[[col]].to_numpy(), dtype=float)
    y = np.asarray(data["score"].to_numpy(), dtype=float)
    gam = LinearGAM(s(0, n_splines=6)).fit(X, y)
    fitted = gam.predict(X)
    return y - fitted


def make_residual_traces(col, data):
    trace_lin = go.Scatter(
        x=data[col],
        y=residuals_linear(col, data),
        mode="markers",
        marker=dict(size=7, color="red"),
        hovertext=data["country"],
        name="Linear",
    )
    trace_gam = go.Scatter(
        x=data[col],
        y=residuals_gam(col, data),
        mode="markers",
        marker=dict(size=7, color="blue"),
        hovertext=data["country"],
        name="GAM",
    )
    return [trace_lin, trace_gam]


# ANOVA table
order = ["gdp", "lifeexp", "support", "freedom", "corruption", "generosity"]
formula = "score ~ " + " + ".join(order)

model = smf.ols(formula=formula, data=happiness).fit()

anova_results = anova_lm(model, typ=1)
resid = model.resid
exog = model.model.exog

white_test = het_white(resid, exog)


# Sequential smooth selection
def sequential_smooth_selection(
    data, y_name, predictors, df_each=6, k=5, aicc_thresh=2
):
    """
    Perform sequential forward selection to decide which predictors
    should be modeled as smooth vs linear in a semiparametric GAM.

    Args:
        data (DataFrame): input dataset
        y_name (str): outcome variable name
        predictors (list of str): candidate predictors
        df_each (int): spline degrees of freedom for each smooth term
        k (int): folds for CV
        aicc_thresh (float): minimum AICc improvement to accept smoothing

    Returns:
        dict with chosen linear predictors, smooth predictors,
        best model, and results
    """

    def aicc(res, n):
        k_params = res.df_model + 1  # includes intercept
        return res.aic + (2 * k_params * (k_params + 1)) / max(n - k_params - 1, 1)

    def fit_gam(subdata, y_name, linear_vars, smooth_vars):
        y = subdata[y_name].values
        X_lin = sm.add_constant(subdata[linear_vars], has_constant="add")
        if len(smooth_vars) == 0:
            return sm.GLM(y, X_lin).fit()
        bs = BSplines(
            subdata[smooth_vars],
            df=[df_each] * len(smooth_vars),
            degree=[3] * len(smooth_vars),
        )
        return GLMGam(y, exog=X_lin, smoother=bs).fit()

    def cv_rmse(model_maker, subdata, y_name, linear_vars, smooth_vars):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        y = subdata[y_name].values
        rmses = []
        for train_idx, test_idx in kf.split(subdata):
            dtr, dte = subdata.iloc[train_idx], subdata.iloc[test_idx]
            m = model_maker(dtr, y_name, linear_vars, smooth_vars)
            if len(smooth_vars) == 0:
                yhat = m.predict(
                    exog=sm.add_constant(dte[linear_vars], has_constant="add")
                )
            else:
                yhat = m.predict()
            rmses.append(np.sqrt(mean_squared_error(dte[y_name].values, yhat)))
        return float(np.mean(rmses))

    # start with all linear
    linear_vars = predictors.copy()
    smooth_vars = []

    n = len(data)
    current = fit_gam(data, y_name, linear_vars, smooth_vars)
    best_aicc = aicc(current, n)
    best_cv = cv_rmse(fit_gam, data, y_name, linear_vars, smooth_vars)

    improved = True
    while improved:
        improved = False
        best_candidate = None
        best_candidate_fit = None
        for v in linear_vars:
            trial_lin = [x for x in linear_vars if x != v]
            trial_smooth = smooth_vars + [v]
            m = fit_gam(data, y_name, trial_lin, trial_smooth)
            m_aicc = aicc(m, n)
            if (best_aicc - m_aicc) >= aicc_thresh:
                best_candidate = v
                best_candidate_fit = m
                best_candidate_aicc = m_aicc

        if best_candidate is not None:
            linear_vars.remove(best_candidate)
            smooth_vars.append(best_candidate)
            current = best_candidate_fit
            best_aicc = best_candidate_aicc
            improved = True

    return {
        "linear_vars": linear_vars,
        "smooth_vars": smooth_vars,
        "best_model": current,
        "best_aicc": best_aicc,
        "best_cv_rmse": best_cv,
    }


sss_result = sequential_smooth_selection(
    data=happiness,
    y_name="score",
    predictors=["gdp", "lifeexp", "support", "freedom", "corruption", "generosity"],
)


# Full model comparison
def fit_linear_model(data, y_name, predictors):
    """
    Fit an OLS regression using statsmodels formula interface.

    Args:
        data (DataFrame): dataset
        y_name (str): outcome variable
        predictors (list of str): predictor variable names

    Returns:
        fitted statsmodels OLS model
    """
    formula = f"{y_name} ~ " + " + ".join(predictors)
    return smf.ols(formula=formula, data=data).fit()


ols_model = fit_linear_model(
    happiness,
    y_name="score",
    predictors=["gdp", "lifeexp", "support", "freedom", "corruption"],
)


def fit_semiparametric_model(
    data, y_name, linear_vars, smooth_vars, df_each=6, degree_each=3
):
    """
    Fit a semi-parametric GAM with specified linear and smooth predictors.

    Args:
        data (DataFrame): dataset
        y_name (str): outcome variable
        linear_vars (list of str): variables treated as linear
        smooth_vars (list of str): variables treated as smooth (splines)
        df_each (int): number of spline basis functions per smooth var
        degree_each (int): degree of spline (default cubic)

    Returns:
        fitted GLMGam model
    """
    y = data[y_name].values
    X_lin = sm.add_constant(data[linear_vars], has_constant="add")

    if len(smooth_vars) > 0:
        bs = BSplines(
            data[smooth_vars],
            df=[df_each] * len(smooth_vars),
            degree=[degree_each] * len(smooth_vars),
        )
        return GLMGam(y, exog=X_lin, smoother=bs).fit()
    else:
        # fall back to GLM if no smooth terms
        return sm.GLM(y, X_lin).fit()


gam_model = fit_semiparametric_model(
    happiness,
    y_name="score",
    linear_vars=["freedom"],  # parametric part
    smooth_vars=["support", "lifeexp", "gdp"],  # nonparametric part
    df_each=6,
    degree_each=3,
)
