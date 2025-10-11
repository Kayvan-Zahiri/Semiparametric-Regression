from statsmodels.graphics.gofplots import qqplot
from dash import dcc, html, Dash, Input, Output
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.formula.api as smf
from section2 import *
from section3 import *


matplotlib.use("Agg")

app = Dash(__name__, assets_folder="assets")
app.title = "Semi-Parametric Regression Visualizer"

# ----------------------
# Dataset
boston = pd.read_csv("./data/boston_housing.csv")


# ----------------------
# Section 1
boston_mapping = {
    "rm": "Average Number of Rooms",
    "crim": "Crime Rate Per Capita",
    "nox": "Nitric Oxides Concentration (pp10m)",
    "age": "Proportion of Units Built Prior to 1940",
    "dis": "Distance from Employment Centers",
    "rad": "Accessibility to Highways",
    "tax": "Property Tax Rate",
    "ptratio": "Pupil-Teacher Ratio",
}
boston_options = [
    {"label": boston_mapping[key], "value": key} for key in boston_mapping
]
boston_mlr = smf.ols(
    "medv ~ crim + dis + nox + rm + age + rad + tax + ptratio", boston
).fit()


# ----------------------
# App layout
app.layout = html.Div(
    [
        html.Header(
            [
                html.H1("Semiparametric Regression"),
                html.P("A Deep Dive"),
                html.Div(
                    "Paul Losiewicz | Vu Chu-Le | Kayvan Zahiri",
                    className="author-info",
                ),
            ]
        ),
        html.Nav(
            html.Ul(
                [
                    html.Li(html.A("Introduction", href="#introduction")),
                    html.Li(html.A("Foundations & Motivation", href="#motivation")),
                    html.Li(html.A("Methods", href="#method")),
                    html.Li(html.A("Applications & Extensions", href="#application")),
                    html.Li(html.A("References", href="#references")),
                ]
            )
        ),
        html.Div(
            className="container blog-section",
            children=[
                html.Div(
                    className="blog-content",
                    children=[
                        html.Section(
                            [
                                html.H2("Introduction", id="introduction"),
                                html.P(
                                    "Selecting the right regression model can be tough. There’s different things to balance and take into account, like the domain, number of parameters to use and much more. Too many parameters could lead to overfitting whereas too few could result in the loss of important information. A solution to this is semiparametric regression. This approach uses a combination of parametric and nonparametric models."
                                ),
                                html.Div(
                                    [
                                        html.Strong("Key Insight: "),
                                        "Semiparametric regression is the combination of finite-dimensional and infinite-dimensional parametric components, utilizing the strengths of both sides, incorporated into one.",
                                    ],
                                    className="highlight-box",
                                ),
                            ]
                        ),
                        html.Section(
                            [
                                html.H2("Foundations & Motivation", id="motivation"),
                                html.P(
                                    "Now, you may be thinking to yourself, “I don’t even know what is a parametric model or a non-parametric model, let alone semiparametric!” Fret not! We will take things slow and start with the basics."
                                ),
                                html.H3("Parametric Models"),
                                html.P(
                                    "Imagine that you got a job offer in a different city, and now you have to decide which town to move to. You also happen to have saved up enough money for a down payment for your first house (yay!), so you’re interested in how different factors affect the median home price of a town. To explore these relationships, you would need a regression model, the simplest of which is the simple linear regression (SLR). As you probably already know, SLR uses a straight line to model the relationship between a single independent variable and a single dependent variable, expressed by the equation:"
                                ),
                                html.Pre(
                                    dcc.Markdown(
                                        "$$y = \\beta_0 + \\beta_1 x + \\epsilon$$",
                                        mathjax=True,
                                    ),
                                    className="formula-box",
                                ),
                                html.P(
                                    [
                                        "Where:",
                                        dcc.Markdown(
                                            """
       - $y$ is the **dependent** variable
       - $x$ is the **independent** variables
       - $\\beta_0$ is the **intercept**, the estimated value of $y$ when $x$ is zero)
       - $\\beta_1$ is the **slope/coefficient** indicating the change in $y$ for a one-unit increase in $x$. 
       - $\\epsilon$ is the **error** term, representing the variation in y that the model doesn’t explain
       """,
                                            mathjax=True,
                                        ),
                                    ]
                                ),
                                html.P(
                                    "SLR is an example of a parametric model because it has a fixed functional form (we assume that the independent variable and dependent variable has a linear relationship), and it has a finite number of parameters (in this case, two, which are the slope and the intercept)."
                                ),
                                html.P(
                                    "Let’s get back to our scenario with an example. To explore the relationship between the number of rooms and housing price, we can fit a linear regression model."
                                ),
                                html.Div(
                                    className="interactive-panel",
                                    children=[
                                        dcc.Dropdown(
                                            id="boston-parameter",
                                            options=boston_options,
                                            value="rm",
                                        ),
                                        html.Div(
                                            className="visualization-container",
                                            children=[
                                                dcc.Graph(id="slr"),
                                            ],
                                        ),
                                        dcc.Markdown(id="equation", mathjax=True),
                                        html.Div(
                                            className="explanation-box",
                                            children=[
                                                html.H4(
                                                    "\ud83d\udca1 How would you interpret the fitted parameters?"
                                                ),
                                                html.H4(
                                                    "Select different options in the drop-down to see the relationships between different independent variable with house price."
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.P(
                                    "We can even generalize this to the multiple linear regression:"
                                ),
                                html.Pre(
                                    dcc.Markdown(
                                        "$$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\dots + \\beta_n x_n + \\epsilon$$",
                                        mathjax=True,
                                    ),
                                    className="formula-box",
                                ),
                                html.P(
                                    [
                                        "Where:",
                                        dcc.Markdown(
                                            """
       - $y$ is the **dependent** variable
       - $x_1, x_2, ..., x_n$ are the **independent** variables
       - $\\beta_0$ is the **intercept**, the estimated value of $y$ when $x$ is zero)
       - $\\beta_1, \\beta_2, ..., \\beta_n$ are the **coefficients** corresponding to each of the independent variables. 
       - $\\epsilon$ is the **error** term, representing the variation in y that the model doesn’t explain
       """,
                                            mathjax=True,
                                        ),
                                    ]
                                ),
                                html.P(
                                    "Let's fit a multiple linear regression model using all the parameters from above and see how well the model fits."
                                ),
                                html.Pre(
                                    children=boston_mlr.summary().as_text(),
                                    className="formula-box",
                                ),
                                html.P(
                                    "Wow, looks like all the variables included are significant predictors of house price!"
                                ),
                                html.H3("⚠️ Hold up!!!! ⚠️"),
                                html.P(
                                    "The words of your favorite professor echoed:  “There's a time and place for everything, but not now. A model is only as good as its assumption!”"
                                ),
                                html.P(
                                    [
                                        "Let's review the assumptions of linear regression and see if they hold for the predictors.",
                                        dcc.Markdown(
                                            """
        - **Linearity**: $x$ and $y$ have a linear relationship with each other 
        - **Normality**: the errors follow a Normal distribution
        - **Homoskedasticity**: the errors have constant variance (the spread of errors doesn't change along the $x$-axis)
        - **Independence**: the errors are independent of each other
        """,
                                            mathjax=True,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    className="interactive-panel",
                                    children=[
                                        html.H3("Let's check if the assumptions hold"),
                                        html.Div(
                                            className="controls-grid",
                                            children=[
                                                html.Div(
                                                    className="control-group",
                                                    children=[
                                                        html.Label("Plot"),
                                                        dcc.Dropdown(
                                                            id="plot",
                                                            options=[
                                                                {
                                                                    "label": "Residuals vs. Fitted",
                                                                    "value": "resid",
                                                                },
                                                                {
                                                                    "label": "QQ-plot",
                                                                    "value": "qq",
                                                                },
                                                            ],
                                                            value="resid",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="control-group",
                                                    children=[
                                                        html.Label("Predictor"),
                                                        dcc.Dropdown(
                                                            id="check-predictor",
                                                            options=boston_options,
                                                            value="rm",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="visualization-container",
                                            children=[
                                                dcc.Graph(id="test"),
                                            ],
                                        ),
                                        html.Div(
                                            className="explanation-box",
                                            children=[
                                                html.H3("\ud83d\udca1 Hint:"),
                                                html.H4(
                                                    html.Ul(id="assumption-comment")
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.P(
                                    "Oh no! Looks like certain assumptions are violated for most of the predictors. If only there is another type of models that allow more flexibility with fewer assumptions!"
                                ),
                                html.P("Well, let me introduce to you..."),
                            ]
                        ),
                        html.Section(
                            [
                                html.H2(
                                    "Linear regression vs Nonparametric Regression",
                                    id="method",
                                ),
                                html.H3("Motivation for GAM"),
                                html.P(
                                    "Nonparametric regression is when there are infinite-dimensional parameters. A key part of this is through the smoothing function. A smoothing function is when the parameters don’t necessarily have a specific functional form, so it’s used as a way to estimate these relationships. The relationships are often non-linear, so having a way to model non-linear relationships is vital. In the real world, it’s challenging to find linear relationships given how much is out of our control. There are too many outside variables and inconsistencies that come into play. So, how do we proceed?"
                                ),
                                html.P(
                                    "One way is through a Generalized Additive Model, or for short, a GAM. This is a common approach to being able to capture nonlinear relationships. It uses the aforementioned smoothing functions to better understand the data. Complex patterns are present throughout, so this approach allows for them to be seen."
                                ),
                                html.H3("GAM vs OLS"),
                                html.P(
                                    "For this deep dive, we’ll be using the same Boston housing dataset from above. The goal is to predict the median housing value (MEDV) from a variety of predictor variables. To list a few, there’s the property tax rate, per capita crime rate, and proportion of old houses, and much more."
                                ),
                                dcc.Markdown(
                                    """
                                    ```python
                                    # We are trying to predict medv, so drop it from the predictors
                                    X = df_boston.drop(columns=['medv'])
                                    y = df_boston['medv']
                                    n = X.shape[0]
                                    
                                    # Add predictors
                                    terms = s(0)
                                    for i in range(1, X.shape[1]):
                                        terms = terms + s(i)

                                    # Build GAM with predictors
                                    gam = LinearGAM(terms).fit(X, y)
                                    ```
                                    """
                                ),
                                dcc.Markdown(
                                    "The next steps are to fit the respective models. After fitting, we then predict and go on to analyze some key values for a potential model selection. The values we’ll be looking at are $R^2$, Adjusted $R^2$, BIC, and AIC.",
                                    mathjax=True,
                                ),
                                html.P(
                                    "It’s important to note that comparing the two models in some aspects can be challenging, and specifically for GAMs due to their infinite parameter set. For OLS, it’s a bit simpler to look at a value like AIC because we know the exact number of coefficients. However, with GAMs, we need things like effective degrees of freedom, splines, and penalties. "
                                ),
                                dcc.Markdown(
                                    """
                                    ```python
                                    # Predictions OLS
                                    y_pred_lr = ols_model.predict(X_lr)

                                    lr_r2 = 0.741
                                    lr_adjr2 = 0.734

                                    p_lr = X.shape[1] + 1

                                    # RSS for BIC
                                    rss_lr = ((y - y_pred_lr)**2).sum()
                                    bic_lr = n * np.log(rss_lr / n) + np.log(n) * p_lr

                                    aic_lr = ols_model.aic


                                    # Predictions GAM
                                    y_pred_gam = gam.predict(X)

                                    gam_r2 = 0.9168

                                    # Effective degrees of freedom
                                    edf = gam.statistics_['edof']
                                    edf_total = edf if np.isscalar(edf) else sum(edf)

                                    # Adjusted R^2
                                    adj_r2_gam = 1 - (1 - gam_r2)*(n - 1)/(n - edf_total - 1)

                                    # BIC approximation
                                    rss_gam = ((y - y_pred_gam)**2).sum()
                                    bic_gam = n * np.log(rss_gam / n) + np.log(n) * edf_total

                                    # AIC and GCV from pyGAM
                                    aic_gam = gam.statistics_['AIC']
                                    gcv_gam = gam.statistics_['GCV']
                                    ```
                                    """
                                ),
                                html.Div(
                                    className="metric-card",
                                    children=[
                                        html.P(
                                            f"GAM R^2: {gam_r2} vs. OLS R^2: {lr_r2}",
                                            className="metric-label",
                                        ),
                                        html.P(
                                            f"GAM Adjusted R^2: {adj_r2_gam} vs. OLS Adjusted R^2: {lr_adjr2}",
                                            className="metric-label",
                                        ),
                                        html.P(
                                            f"GAM BIC (approx): {bic_gam} vs. OLS BIC: {bic_lr}",
                                            className="metric-label",
                                        ),
                                        html.P(
                                            f"GAM AIC: {aic_gam} vs. OLS AIC: {aic_lr}",
                                            className="metric-label",
                                        ),
                                    ],
                                ),
                                dcc.Markdown(
                                    "In analyzing the above results, we can see that GAM are favored in terms of $R^2$, Adjusted $R^2$, and BIC. However, OLS wins for AIC. Why might this be? The GAM has a large effective degrees of freedom count, potentially offsetting the log likelihood improvement. The AIC vs BIC results are somewhat surprising, but from the overall results, GAM is favored. ",
                                    mathjax=True,
                                ),
                                html.P(
                                    "To further dive into this, let’s create an interactive Plotly visualization! For each predictor, we plot the actual data points of the observed house prices vs that predictor, the linear regression prediction, and the GAM prediction. "
                                ),
                                html.Div(
                                    className="interactive-panel",
                                    children=[
                                        html.Div(
                                            className="visualization-container",
                                            children=[
                                                dcc.Graph(id="figure3"),
                                            ],
                                        ),
                                        html.Div(
                                            className="explanation-box",
                                            children=[
                                                html.H4(
                                                    "From the visualizations, we can see some quite similar plots and very different ones. Some predictors with similar blue and red lines are lower status in neighborhood, pupil-teacher ratio, proportion of black residents, and more. The similar ones suggest that there is a linear relationship and that perhaps we should stick with OLS, whereas the others indicate that a GAM is better. One interesting plot that stands out is the age of the house. Does the age not impact the price that much? Maybe, given how there can be an old fixer-upper, or conversely, a Painted Lady. "
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.H3("Tradeoffs"),
                                html.P(
                                    "So, when should we go with linear regression vs something nonparametric, like a GAM, and vice versa? Like all situations…it depends. With linear regression, it’s much simpler to understand and interpret. There are clear values in regards to each predictor and how much they will impact the outcome. However, again, they are less flexible and can miss nonlinear relationships. Big patterns and curves can be missed, but if you’re expecting linear relationships, the choice is most likely OLS."
                                ),
                                html.P(
                                    "For GAMs, a big plus is flexibility. Nonlinear relationships are captured, resulting in better predictions for a lot of real-world situations. However, some big downfalls are interpretability and communication. There isn’t a direct slope or number for each predictor you can point to, indicating how this particular variable affects the outcome. For non-technical audiences that you may be working with, getting your point across becomes a bigger challenge."
                                ),
                                html.P(
                                    "Overall, there are a variety of pros and cons for each type of model. Linear regression models are easier to interpret and understand, but the ability to capture more complex, nonlinear relationships is decreased. Conversely, nonparametric models can have increased accuracy, but are more challenging to interpret and communicate. Deciding which route to go ultimately depends on the domain, your team, and the situation. "
                                ),
                            ]
                        ),
                        html.Section(
                            [
                                html.H2("Applications & Extensions", id="application"),
                                html.H3("Motivation & Use Cases"),
                                html.P(
                                    "So, when is a linear model too simple and rigid to encapsulate the full complexity of a dataset?  Well, quite honestly, that's up to the judgement of the person generating the model."
                                ),
                                dcc.Markdown(
                                    "One of the key assumptions of the OLS linear regression model is that there exhibits a linear relationship between the coefficients $\\beta$ and the response $Y$, and while there are an almost indefinite number of ways to transform a given dataset to meet this assumption, don't you think that this requirement can be a bit rigid?",
                                    mathjax=True,
                                ),
                                html.P(
                                    "Sometimes, in domains such as in medicine and social sciences, the relationships between variables are flexible and influential on one another.  In the comparison of linear (parametric) and non-parametric models, the later does not require any explicit definition of interaction to pick up on such patterns.  If this seems too good to be true, it actually is.  The estimation of these non-parametric functions takes away the clear interpretability offered by linear regression models; all that you're left with is the curve."
                                ),
                                html.P(
                                    "Similarly to parametric regression models too, non-parametric predictors are plagued with the same curse of dimensionality. As we will encounter later in the demo, the flexible parts of the semi-parametric model stich together small, preset functions, which are weighted through a loss function to estimate a given distribution.  Each one of these functions adds a degree of freedom, resulting in overfitting, a loss of statistical power, and an increase in computational cost."
                                ),
                                html.P(
                                    "Below we are going to do a deep-dive demo, demonstrating the difference between linear and non-parametric models, and the benefits of using them both in a semi-parametric regression model."
                                ),
                                html.H3(
                                    "Single-Variable Fits: Linear vs. Semi-Parametric"
                                ),
                                html.P(
                                    "We'll start this demo by introducing our dataset.  We will be using the 2019 World Happiness Report, which essentially asks people around the world how happy they are on a scale of 1-10.  The predictor variables, as listed below, are metrics meant to explain the impact on the average happiness levels in each respective country.  There is plenty of opportunity for interactive impacts on this dataset, along with the natural complexities of quantitatively measuring happiness.  I'm personally not aware of any scientifically explained linear relationships between corruption and happiness, making this an interesting problem!"
                                ),
                                dcc.Markdown(
                                    "We're going to start this demo by doing single variable comparisons between each of the potential predictors: `gdp`, `support`, `lifeexp`, `freedom`, `generosity`, and `corruption`, with our response variable, `score`.  Below is an interactive scatter plot demonstrating the differences in regression model fitting between a linear parametric model and a nonlinear non-parametric model.",
                                    mathjax=True,
                                ),
                                html.Div(
                                    className="interactive-panel",
                                    children=[
                                        html.Div(
                                            className="visualization-container",
                                            children=[
                                                dcc.Graph(id="figure4"),
                                            ],
                                        )
                                    ],
                                ),
                                html.P(
                                    "As demonstrated above, it seems that the non-parametric curve is really good on picking up subtle nuances in the data, but as mentioned, there's no rigid or explicit way to mathematically interpret this relationship besides looking at the graph.  In order to pick which predictors may be better fit for a non-parametric curve in a full-model, it's a good idea to see another side-by-side comparison of residuals."
                                ),
                                html.Div(
                                    className="interactive-panel",
                                    children=[
                                        html.Div(
                                            className="visualization-container",
                                            children=[
                                                dcc.Graph(id="figure5"),
                                            ],
                                        ),
                                    ],
                                ),
                                html.P(
                                    "So while the linear residuals don't exhibit any clear patterns of non-linearity, the residuals for the non-parametric model do demonstrate marginally positive differences.  It seems that the residuals are more spread out and symmetric on the right for almost all predictors, but I also want to reiterate that there are no obvious signs of heteroskedasticity on the left.  Taking all of this into account though, this was meant to just be initial expousure to the difference between parametric and non-parametric models.  Of course, variables interact with eachother, and the predictive power of a predictor may change significantly when added to a MLR model.  In the next section, we look to tackle model selection."
                                ),
                                html.H3("ANOVA & Model Selection"),
                                html.P(
                                    "Before we jump the gun and try to fix something that may already work, let's look at a sequential ANOVA test on a simple MLR model predicting happiness."
                                ),
                                html.Pre(
                                    children=str(anova_results),
                                    className="formula-box",
                                ),
                                dcc.Markdown(
                                    "Well, when put in this order, it seems that each predictor explains a significant level of variance on our response, `score`.  Just because one model works though, does not mean another is not necessary.  As mentioned in the residual section, there are no clear signs of blatant non-linearity or heteroskedasticity, but also there are statistical tests meant to measure this sort of thing.  Enter, the White test, which essentially regresses the square of the residuals onto the original regressors, their squares, and their cross-products, and calculates something called the auxilliary regression, which follows a chi-squared distribution with k degrees of freedom.",
                                    mathjax=True,
                                ),
                                html.Div(
                                    className="metric-card",
                                    children=[
                                        html.H4("Het-white results"),
                                        html.P(
                                            f"Lagrange multiplier statistic: {white_test[0]:.5f}",
                                            className="metric-label",
                                        ),
                                        html.P(
                                            f"The p-value of lagrange multiplier test.: {white_test[1]:.5f}",
                                            className="metric-label",
                                        ),
                                        html.P(
                                            f"The F-statistic of the hypothesis that the error variance does not depend on x.: {white_test[2]:.5f}",
                                            className="metric-label",
                                        ),
                                        html.P(
                                            f"The p-value for the F-statistic: {white_test[3]}",
                                            className="metric-label",
                                        ),
                                    ],
                                ),
                                html.P(
                                    "The second test-statistic is an F-test, which is the usual statistic for auxilliary regression.  It is meant to test whether extra regressors significantly explain the squared residuals, and with an incredibly low p-value, our Het White test suggests that there is significant heteroskedasticity in this dataset.  I do find it worthy to outline that there are patterns of increased Type-I error in smaller datasets, but even so, the significant results make it at least a try to fit a semi-parametric model onto this dataset."
                                ),
                                html.P(
                                    "But the quesiton that remains is, how do we even know which predictors to fit as parametric and which to fit as non-parametric?  In the case of this model, we're going to use a process where we assume every predictor is linear, and one by one, we test whether a predictor being non-parametric adds the necessary amount of predictive power, while keeping model complexity into account.  We can continue this cycle until there's no further improvements."
                                ),
                                html.Pre(
                                    children=[
                                        f"Linear predictors: {sss_result["linear_vars"]}\n",
                                        f"Smooth predictors: {sss_result["smooth_vars"]}\n\n",
                                        sss_result["best_model"].summary().as_text(),
                                    ],
                                    className="formula-box",
                                ),
                                dcc.Markdown(
                                    "As we can see, the algorithm decided that `freedom`, `corruption`, and `generosity` are best suited as linear predictors, while `support`, `lifeexp`, and `gdp` were chosen to be smooth.  If we take a deeper dive into the table, each smooth predictor has 6 functions associated with it.  These functions are called `B-splines`.  They are essentiallly independent, pre-determined functions which are weighted to find the optimal function approximation between a given predictor and the response.  When a p-value is insignificant, it is just suggesting that this specific spline has no significant impact on approximating the curve. ",
                                    mathjax=True,
                                ),
                                html.P(
                                    "In our problem, we chose 6 basis functions for each non-parametric parameter.  It is general practice to start with between 5 and 10 per predictor, and we thought 6 basis functions provides enough to take account for subtle nuances while keeping the degrees of freedom in check.  While there are some limited options to go through hyper-parameter tuning, such as using metrics like AIC/BIC, GAM models already use their own penalties to determine which and how many basis functions to include in the final model.  The number of basis functions you intially provide is essentially a safe upper bound for your model to converge from."
                                ),
                                dcc.Markdown(
                                    "We can also see that two of the predictors that were chosen to be linear, `corruption` and `generosity`, did not provide any sort of additional explanation when predicting happiness.  As a result, we will omit these predictors from our final model.",
                                    mathjax=True,
                                ),
                                html.H3("Full Model Comparison"),
                                dcc.Markdown(
                                    "In order to truly gauge the effectiveness of this final mixed, semi-parametric model, I found it necessary to fit a simple MLR model using all of our predictors, besides `generosity`, which did not provide the necessary predictive power.",
                                    mathjax=True,
                                ),
                                html.Pre(
                                    children=[ols_model.summary().as_text()],
                                    className="formula-box",
                                ),
                                html.Pre(
                                    children=[gam_model.summary().as_text()],
                                    className="formula-box",
                                ),
                                html.P(
                                    "In order to compare these two models, we will use AIC and BIC, which both have their own penalties that should take into account the additional degrees of freedom introduced by the semi-parametric model."
                                ),
                                html.Div(
                                    className="metric-card",
                                    children=[
                                        html.P(
                                            f"OLS AIC: {ols_model.aic}, BIC: {ols_model.bic}",
                                            className="metric-label",
                                        ),
                                        html.P(
                                            f"GAM AIC: {gam_model.aic}, BIC: {gam_model.bic}",
                                            className="metric-label",
                                        ),
                                    ],
                                ),
                                html.P(
                                    "Interestingly, while AIC performs slightly better in our semi-parametric model than the full linear model, the BIC for the semi-parametric model came back negative, and not just negative, but *very* negative.  This was originally a quite shocking result, especially considering that BIC is supposed to penalize models more harshly for model complexity.  I deduce that the incredibly low metric numbers for our BIC are due to the ability for non-parametric variables to pick up on nuances that are simply not possible in a simple MLR model, almost similar to a deep neural network.  Despite these significant results, I want to finish this demo with a discussion on the practicality of semi-parametric models."
                                ),
                                html.H3("Takeaways"),
                                html.P(
                                    "Although we were initially excited by the results of this demo, we have to ground ourselves and consider the very real tradeoffs when dealing with semi-parametric and non-parametric regression models."
                                ),
                                dcc.Markdown(
                                    "In the specific case of this demo, it was just unrealistic to perform k-fold cross validation to test whether the semi-parametric model was potentially overfitting the data.  The original **Happiness Report** only included approximately 150 data points, and maybe our semi-parametric model was just able to pick up on every single nuance and interaction between the predictors. "
                                ),
                                dcc.Markdown(
                                    "Additionally, the loss of interpretability cannot be over-stated.  One of the biggest benefits to the linear regression model, despite its simplicities, is you see *exactly* what is going on under the hood.  The coefficient interpretations provide isolated and clear takeaways that can be demonstrated to a non-technical audience.  Like almost all model-selection discussions, the choice between semi-parametric and linear models comes down to your purpose."
                                ),
                                dcc.Markdown(
                                    "*Does interpretation even add anything?*"
                                ),
                                html.P(
                                    "In the case of scientific or economic forecasting, the main goal is coming up with models that predict accuratelyl; what's specifically changing the response variable does not necessarily matter."
                                ),
                                html.P(
                                    "In the specific case of this happiness dataset, the unit-to-unit interpretation between GDP and Happines Score, an otherwise arbitrary metric, seems miniscule.  But sometimes, statistically strong results regarding linear relationships can be particularly powerful, insightful, and more important than the actual results of a predictive model."
                                ),
                                html.P(
                                    "In the future, our group would love the opportunity to further explore the performance of these semi-parametric models in vastly different datasets and domains, and to see whether these models overfit as much as they are said to."
                                ),
                            ]
                        ),
                        html.Section(
                            [
                                html.H2("References", id="references"),
                                html.Ol(
                                    [
                                        html.Li(
                                            dcc.Markdown(
                                                "[https://discdown.org/flexregression/smoothreg.html](https://discdown.org/flexregression/smoothreg.html)"
                                            )
                                        ),
                                        html.Li(
                                            dcc.Markdown(
                                                "[https://www.kaggle.com/datasets/unsdsn/world-happiness](https://www.kaggle.com/datasets/unsdsn/world-happiness)"
                                            )
                                        ),
                                        html.Li(
                                            dcc.Markdown(
                                                "[https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data](https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data)"
                                            )
                                        ),
                                    ],
                                    className="reference-list",
                                ),
                            ]
                        ),
                    ],
                )
            ],
        ),
        # html.Footer(
        #     className="footer",
        # ),
    ]
)


# ----------------------
# Callbacks
@app.callback(
    Output("slr", "figure"),
    Output("equation", "children"),
    Input("boston-parameter", "value"),
)
def figure_1(x):
    label = boston_mapping[x]
    fig = px.scatter(boston, x=x, y="medv", trendline="ols")
    fig.update_traces(
        marker=dict(size=6, color="gray", opacity=0.6), selector=dict(mode="markers")
    )
    fig.update_traces(line=dict(color="red"), selector=dict(mode="lines"))
    fig.update_layout(
        title={
            "text": f"{label} Impact on Median House Price",
            "x": 0.5,
            "xanchor": "center",
        },
        width=900,
        height=700,
        hovermode="closest",
    )
    results = px.get_trendline_results(fig)
    model_results = results.px_fit_results.iloc[0]
    slope = model_results.params[1]
    intercept = model_results.params[0]
    equation = (
        f"Equation of the fitted model is: $$y = {intercept:.2f} + {slope:.2f}x$$"
    )
    return fig, equation


@app.callback(
    Output("test", "figure"),
    Output("assumption-comment", "children"),
    Input("plot", "value"),
    Input("check-predictor", "value"),
)
def figure_2(plot, x):
    if plot == "resid":
        model = smf.ols(f"medv ~ {x}", boston).fit()
        fig = px.scatter(x=model.fittedvalues, y=model.resid)
        fig.update_traces(
            marker=dict(size=6, color="gray", opacity=0.6),
            selector=dict(mode="markers"),
        )
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
        fig["layout"].update(
            {
                "title": f"Residuals vs. Fitted Values Plot: {boston_mapping[x]}",
                "xaxis": {"title": "Residuals"},
                "yaxis": {"title": "Sample Quantities"},
                "showlegend": False,
                "width": 900,
                "height": 850,
            }
        )
        comment = [
            html.Li(
                "Do the residuals look randomly distributed around the 0 line? If yes, the independent and dependent variable has a linear relationship."
            ),
            html.Li(
                "Do the residuals roughly form a horizontal band around the 0 line? If yes, the error terms have roughly constant variance."
            ),
            html.Li(
                "Is there a clear pattern of the points in the scatter plot? If no, the error terms are independent."
            ),
        ]
    else:
        gauss_data = boston[x]
        qqplot_data = qqplot(gauss_data, line="s").gca().lines

        fig = go.Figure()

        fig.add_trace(
            {
                "type": "scatter",
                "x": qqplot_data[0].get_xdata(),
                "y": qqplot_data[0].get_ydata(),
                "mode": "markers",
                "marker": {"color": "gray"},
            }
        )

        fig.add_trace(
            {
                "type": "scatter",
                "x": qqplot_data[1].get_xdata(),
                "y": qqplot_data[1].get_ydata(),
                "mode": "lines",
                "line": {"color": "blue"},
            }
        )

        fig["layout"].update(
            {
                "title": "Quantile-Quantile Plot",
                "xaxis": {"title": "Theoretical Quantities"},
                "yaxis": {"title": "Sample Quantities"},
                "showlegend": False,
                "width": 900,
                "height": 850,
            }
        )
        comment = [
            html.Li(
                "Do the points fall along a straight line? If yes, the error terms are Normally distributed."
            )
        ]
    return fig, comment


@app.callback(Output("figure3", "figure"), Input("figure3", "id"))
def figure_3(_):
    predictors = boston_X.columns.tolist()
    n_predictors = len(predictors)

    fig = go.Figure()

    for i, predictor in enumerate(predictors):
        x_vals = np.linspace(boston_X[predictor].min(), boston_X[predictor].max(), 100)

        # Linear regression line for predictor
        beta_0 = b_ols_model.params.iloc[0]
        beta_i = b_ols_model.params.iloc[i + 1]
        y_pred_lr_line = beta_0 + beta_i * x_vals

        # GAM smooth for predictor
        X_grid = boston_X.mean().values.reshape(1, -1).repeat(100, axis=0)
        X_grid[:, i] = x_vals
        y_pred_gam_line = gam.predict(X_grid)

        # Scatter points (actual house prices)
        fig.add_trace(
            go.Scatter(
                x=boston_X[predictor],
                y=boston_y,
                mode="markers",
                name="Actual Prices",
                visible=(i == 0),
                marker=dict(color="lightgrey"),
                hovertemplate=f"{friendly_labels.get(predictor,predictor)}: %{{x}}<br>Price: $%{{y}}k",
            )
        )

        # Linear regression line
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_pred_lr_line,
                mode="lines",
                name="Simple Linear Prediction",
                visible=(i == 0),
                line=dict(color="blue", width=3),
            )
        )

        # GAM smooth line
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_pred_gam_line,
                mode="lines",
                name="Flexible Prediction",
                visible=(i == 0),
                line=dict(color="red", width=3),
            )
        )

    # Dropdown menu
    buttons = []
    for i, predictor in enumerate(predictors):
        visible = [False] * n_predictors * 3
        visible[i * 3 : i * 3 + 3] = [True, True, True]

        # Friendly display name for title/axes
        display_name = friendly_labels.get(predictor, predictor)

        buttons.append(
            dict(
                label=predictor.lower(),
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": f"House Price vs {display_name}",
                        "xaxis": {"title": display_name},
                        "yaxis": {"title": "House Price ($1000s)"},
                    },
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(active=0, buttons=buttons, x=1.05, y=1, xanchor="right", yanchor="top")
        ],
        title=f"House Price vs {friendly_labels.get(predictors[0], predictors[0])}",
        xaxis_title=friendly_labels.get(predictors[0], predictors[0]),
        yaxis_title="House Price ($1000s)",
        legend=dict(
            x=0.02,
            y=0.999,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=1,
        ),
        width=900,
        height=600,
    )

    return fig


@app.callback(Output("figure4", "figure"), Input("figure4", "id"))
def build_figure(_):
    data = happiness
    initial_x = "gdp"
    # --- Build subplots ---
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Linear Fit", "Smooth (GAM) Fit")
    )

    # initial traces
    traces = make_traces(initial_x, data)
    fig.add_trace(traces[0], row=1, col=1)
    fig.add_trace(traces[1], row=1, col=2)
    fig.add_trace(traces[2], row=1, col=1)
    fig.add_trace(traces[3], row=1, col=2)

    # --- Dropdown updates ---
    buttons = []
    for col in x_options:
        traces = make_traces(col, data)
        buttons.append(
            dict(
                label=col,
                method="update",
                args=[
                    {
                        "x": [traces[0].x, traces[1].x, traces[2].x, traces[3].x],
                        "y": [traces[0].y, traces[1].y, traces[2].y, traces[3].y],
                    },
                    {
                        "title.text": f"Model Fit Comparison: {col}",
                        "xaxis.title.text": col,
                        "xaxis2.title.text": col,
                        "yaxis.title.text": "Happiness Score",
                        "yaxis2.title.text": "Happiness Score",
                    },
                ],
            )
        )

    fig.update_layout(
        title=f"Model Fit Comparison: {initial_x}",
        width=1000,
        height=500,
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=0.5,
                y=1.15,
                xanchor="center",
                yanchor="top",
            )
        ],
    )
    return fig


@app.callback(Output("figure5", "figure"), Input("figure5", "id"))
def build_residual_figure(_):
    data = happiness
    initial_x = "gdp"
    # --- Build subplots ---
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Linear Residuals", "Smooth (GAM) Residuals")
    )

    # initial traces
    traces = make_residual_traces(initial_x, data)
    fig.add_trace(traces[0], row=1, col=1)  # linear panel
    fig.add_trace(traces[1], row=1, col=2)  # gam panel

    # horizontal zero lines
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

    # dropdown buttons
    buttons = []
    for col in x_options:
        traces = make_residual_traces(col, data)
        buttons.append(
            dict(
                label=col,
                method="update",
                args=[
                    {
                        "x": [traces[0].x, traces[1].x],
                        "y": [traces[0].y, traces[1].y],
                        "hovertext": [traces[0].hovertext, traces[1].hovertext],
                    },
                    {
                        "title.text": f"Residual Comparison: {col}",
                        "xaxis.title.text": col,
                        "xaxis2.title.text": col,
                        "yaxis.title.text": "Residuals",
                        "yaxis2.title.text": "Residuals",
                    },
                    [0, 1],  # update both traces only, not hlines
                ],
            )
        )

    fig.update_layout(
        title=f"Residual Comparison: {initial_x}",
        width=1000,
        height=500,
        showlegend=False,
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=0.5,
                xanchor="center",
                y=1.15,
                yanchor="top",
            )
        ],
    )
    return fig


if __name__ == "__main__":
    app.run(port=8050)
