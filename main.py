import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objects as go
import plotly.express as px

app = dash.Dash(__name__, assets_folder="assets")
app.title = "Semi-Parametric Regression Visualizer"


def generate_data(n_samples=200, n_features=10, noise=0.5, random_state=1):
    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        coef=True,
        random_state=random_state,
    )
    return X, y, coef


def fit_model(method, alpha, l1_ratio, X_train, y_train, X_test):
    # alpha -> regularization strength (lambda in your HTML)
    if method == "ridge":
        model = Ridge(alpha=alpha)
    elif method == "lasso":
        model = Lasso(alpha=alpha, max_iter=10000)
    elif method == "elastic":
        # l1_ratio between 0 (Ridge) and 1 (Lasso)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds


# Compute coefficient path for a given method over a grid of alphas
def coefficient_path(method, alphas, l1_ratio, X, y):
    coefs = []
    for a in alphas:
        if method == "ridge":
            m = Ridge(alpha=a)
        elif method == "lasso":
            m = Lasso(alpha=a, max_iter=10000)
        elif method == "elastic":
            m = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=10000)
        else:
            m = LinearRegression()
        m.fit(X, y)
        try:
            coef = m.coef_
        except Exception:
            coef = np.zeros(X.shape[1])
        coefs.append(coef)
    coefs = np.array(coefs)  # shape: (len(alphas), n_features)
    return coefs


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
                    html.Li(html.A("Theory", href="#theory")),
                    html.Li(html.A("Comparison", href="#comparison")),
                    html.Li(html.A("Interactive App", href="#interactive")),
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
                        html.Span("BLOG POST", className="section-tag"),
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
                                html.H2("Theoretical Foundation", id="theory"),
                                html.H3("Mathematical Framework"),
                                html.P(
                                    "A common penalized loss used in regularized regression:"
                                ),
                                html.Pre(
                                    "L(\u03b2) = ||y - X\u03b2||^2 + \u03bb \u00b7 P(\u03b2)",
                                    className="formula-box",
                                ),
                                html.H3("Key Concepts"),
                                html.Ul(
                                    [
                                        html.Li("Concept one: Bias-variance tradeoff"),
                                        html.Li(
                                            "Concept two: Regularization encourages simpler models"
                                        ),
                                        html.Li(
                                            "Concept three: Elastic Net mixes L1 and L2 penalties"
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        html.Section(
                            [
                                html.H2("Comparative Analysis", id="comparison"),
                                html.H3("Method Comparison"),
                                html.Pre(
                                    "Ridge: ||\u03b2||_2^2   |   Lasso: ||\u03b2||_1   |   Elastic Net: \u03b1||\u03b2||_1 + (1-\u03b1)||\u03b2||_2^2",
                                    className="formula-box",
                                ),
                            ]
                        ),
                        html.Section(
                            [
                                html.H2("References", id="references"),
                                html.Ol(
                                    [
                                        html.Li(
                                            'Author, A. (2024). "Lorem Ipsum Dolor Sit Amet." Journal of Statistical Learning, 45(3), 123-145.'
                                        ),
                                        html.Li(
                                            'Smith, B. & Jones, C. (2023). "Consectetur Adipiscing Elit." Machine Learning Review, 12(2), 67-89.'
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
        html.Div(
            className="app-section",
            id="interactive",
            children=[
                html.Div(
                    className="app-container",
                    children=[
                        html.Div(
                            className="app-header",
                            children=[
                                html.Span(
                                    "INTERACTIVE APPLICATION", className="section-tag"
                                ),
                                html.H2("Regularization Explorer"),
                                html.P(
                                    "Adjust the parameters below to visualize how different regularization techniques affect regression models"
                                ),
                            ],
                        ),
                        html.Div(
                            className="interactive-panel",
                            children=[
                                html.H3("Control Panel"),
                                html.Div(
                                    className="controls-grid",
                                    children=[
                                        html.Div(
                                            className="control-group",
                                            children=[
                                                html.Label("Regularization Method"),
                                                dcc.Dropdown(
                                                    id="method",
                                                    options=[
                                                        {
                                                            "label": "Ridge Regression (L2)",
                                                            "value": "ridge",
                                                        },
                                                        {
                                                            "label": "Lasso Regression (L1)",
                                                            "value": "lasso",
                                                        },
                                                        {
                                                            "label": "Elastic Net",
                                                            "value": "elastic",
                                                        },
                                                        {
                                                            "label": "No Regularization",
                                                            "value": "none",
                                                        },
                                                    ],
                                                    value="ridge",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="control-group",
                                            children=[
                                                html.Label(
                                                    [
                                                        "Lambda (\u03bb): ",
                                                        html.Span(
                                                            id="lambdaValue",
                                                            children="1.0",
                                                            className="slider-value",
                                                        ),
                                                    ]
                                                ),
                                                dcc.Slider(
                                                    id="lambda",
                                                    min=0.0,
                                                    max=10.0,
                                                    step=0.1,
                                                    value=1.0,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="control-group",
                                            children=[
                                                html.Label(
                                                    [
                                                        "Number of Features: ",
                                                        html.Span(
                                                            id="featuresValue",
                                                            children="10",
                                                            className="slider-value",
                                                        ),
                                                    ]
                                                ),
                                                dcc.Slider(
                                                    id="features",
                                                    min=2,
                                                    max=50,
                                                    step=1,
                                                    value=10,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="control-group",
                                            children=[
                                                html.Label(
                                                    [
                                                        "Noise Level: ",
                                                        html.Span(
                                                            id="noiseValue",
                                                            children="0.5",
                                                            className="slider-value",
                                                        ),
                                                    ]
                                                ),
                                                dcc.Slider(
                                                    id="noise",
                                                    min=0.0,
                                                    max=2.0,
                                                    step=0.1,
                                                    value=0.5,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="visualization-container",
                                    children=[dcc.Graph(id="main-scatter")],
                                ),
                                html.Div(
                                    className="metrics-grid",
                                    children=[
                                        html.Div(
                                            className="metric-card",
                                            children=[
                                                html.Div(
                                                    "Training R²",
                                                    className="metric-label",
                                                ),
                                                html.Div(
                                                    id="train-r2",
                                                    className="metric-value",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="metric-card",
                                            children=[
                                                html.Div(
                                                    "Test R²", className="metric-label"
                                                ),
                                                html.Div(
                                                    id="test-r2",
                                                    className="metric-value",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="metric-card",
                                            children=[
                                                html.Div(
                                                    "MSE", className="metric-label"
                                                ),
                                                html.Div(
                                                    id="mse", className="metric-value"
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="metric-card",
                                            children=[
                                                html.Div(
                                                    "Non-Zero Coefficients",
                                                    className="metric-label",
                                                ),
                                                html.Div(
                                                    id="nonzero",
                                                    className="metric-value",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="explanation-box",
                                    children=[
                                        html.H4("\ud83d\udca1 What's happening here?"),
                                        html.P(
                                            "Adjust the parameters above to see how the model changes. The plots and metrics update automatically."
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="interactive-panel",
                            children=[
                                html.H3("Coefficient Path Visualization"),
                                html.Div(
                                    className="visualization-container",
                                    children=[dcc.Graph(id="coef-path")],
                                ),
                                html.Div(
                                    className="explanation-box",
                                    children=[
                                        html.H4(
                                            "\ud83d\udca1 Understanding Coefficient Paths"
                                        ),
                                        html.P(
                                            "This visualization shows how coefficients change as regularization strength increases. Notice how different regularization methods affect the coefficient trajectories differently."
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
        html.Footer(
            className="footer",
            children=[
                html.P("\u00a9 2025 [Your Name] | MSDS Portfolio Project"),
                html.P("Built with Dash + scikit-learn"),
            ],
        ),
    ]
)

# ----------------------
# Callbacks


@app.callback(
    Output("lambdaValue", "children"),
    Output("featuresValue", "children"),
    Output("noiseValue", "children"),
    Input("lambda", "value"),
    Input("features", "value"),
    Input("noise", "value"),
)
def update_slider_labels(lam, feats, noise):
    return f"{lam:.1f}", str(int(feats)), f"{noise:.1f}"


@app.callback(
    Output("main-scatter", "figure"),
    Output("train-r2", "children"),
    Output("test-r2", "children"),
    Output("mse", "children"),
    Output("nonzero", "children"),
    Input("method", "value"),
    Input("lambda", "value"),
    Input("features", "value"),
    Input("noise", "value"),
)
def update_main_plot(method, lam, features, noise):
    # Generate synthetic data
    X, y, true_coef = generate_data(
        n_samples=300, n_features=int(features), noise=float(noise), random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Fit model
    l1_ratio = 0.5
    model, preds = fit_model(
        method,
        alpha=float(lam),
        l1_ratio=l1_ratio,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
    )
    train_preds = model.predict(X_train)

    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, preds)
    mse_val = mean_squared_error(y_test, preds)
    nonzero = np.sum(np.abs(model.coef_) > 1e-6) if hasattr(model, "coef_") else 0

    # Build a 2D projection to visualize: use first feature vs response for scatter
    if X.shape[1] >= 2:
        scatter_x = X_test[:, 0]
    else:
        scatter_x = np.arange(len(y_test))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scatter_x,
            y=y_test,
            mode="markers",
            name="Actual",
            marker=dict(opacity=0.7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scatter_x,
            y=preds,
            mode="markers",
            name="Predicted",
            marker=dict(opacity=0.9),
        )
    )
    fig.update_layout(
        title="Actual vs Predicted (projection on first feature)",
        xaxis_title="Feature Value (first column)",
        yaxis_title="Response",
    )

    # Format metrics nicely
    return fig, f"{train_r2:.2f}", f"{test_r2:.2f}", f"{mse_val:.2f}", str(int(nonzero))


@app.callback(
    Output("coef-path", "figure"),
    Input("method", "value"),
    Input("features", "value"),
    Input("noise", "value"),
)
def update_coef_path(method, features, noise):
    # small dataset for coefficient path
    X, y, _ = generate_data(
        n_samples=200, n_features=int(features), noise=float(noise), random_state=0
    )
    alphas = np.logspace(-3, 1.5, 60)
    coefs = coefficient_path(
        method, alphas, l1_ratio=0.5, X=X, y=y
    )  # shape (len(alphas), n_features)

    fig = go.Figure()
    for feat_idx in range(coefs.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=np.log10(alphas),
                y=coefs[:, feat_idx],
                mode="lines",
                name=f"Coef {feat_idx+1}",
                opacity=0.8,
            )
        )

    fig.update_layout(
        title="Coefficient paths vs log10(lambda)",
        xaxis_title="log10(lambda)",
        yaxis_title="Coefficient value",
        showlegend=False,
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
