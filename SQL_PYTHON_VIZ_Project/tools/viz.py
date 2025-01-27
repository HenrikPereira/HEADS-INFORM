import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import plotly.express as px
import pingouin as pg
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.ensemble import IsolationForest

def plot_histogram_qq(df: pd.DataFrame, feature: str, transform: str = None):
    """
    Plot a histogram and QQ plot for a given feature.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature (str): Feature to plot.
        transform (str): Transformation to apply (log, zscore, minmax).
    """
    if transform == "log":
        df[feature] = np.log(df[feature].replace(0, 1e-10))
    elif transform == "zscore":
        df[feature] = pg.zscore(df[feature])
    elif transform == "minmax":
        df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df[feature], kde=True, ax=axes[0])
    pg.qqplot(df[feature], ax=axes[1], square=False)

    axes[0].set_title(f"{feature} Histogram")
    axes[1].set_title(f"{feature} QQ Plot")
    st.pyplot(fig)


def plot_ridgeline_plot_seaborn_alt(df: pd.DataFrame, features: list[str], name: str, transform: str = None):
    """
    Gera um gráfico tipo ridgeline com densidade usando Seaborn.
    Adaptações: Aplica a escala log no eixo x compartilhado.
    """
    darkgreen = '#9BC184'
    midgreen = '#C2D6A4'
    lightgreen = '#E7E5CB'
    darkgrey = '#525252'
    colors = [lightgreen, midgreen, darkgreen, midgreen, lightgreen]

    n_features = len(features)

    log_scale = True if transform == 'log' else False

    if transform == 'zscore':
        df = df.copy()
        for feature in features:
            df[feature] = scipy.stats.zscore(df[feature])
        min_val = df[features].min().min()
        xlabel_aid = ' (Z-score)'
    elif transform == 'log':
        df = df.copy()
        min_val = df[features].replace(0, np.nan).min().min()  # Eliminar zeros para evitar log(0)
        xlabel_aid = ' (Log scale)'
    elif transform == 'minmax':
        df = df.copy()
        for feature in features:
            df[feature] = sklearn.preprocessing.minmax_scale(df[feature])
        min_val = df[features].min().min()
        xlabel_aid = ' (MinMax scale)'
    else:
        min_val = df[features].min().min()
        xlabel_aid = ''

    max_val = df[features].max().max()

    # Calculate the mean for each column
    column_means = df[features].mean()

    # Sort the column names by the mean values
    sorted_features = column_means.sort_values(ascending=False).index

    fig, axes = plt.subplots(nrows=n_features, ncols=1, figsize=(10, 6), sharex=True, sharey=False)
    axes = axes.flatten()  # Garantir que os eixos sejam indexáveis para iteração

    if log_scale:
        values = np.logspace(np.log10(min_val), np.log10(max_val), 7).round(3)
    else:
        values = np.linspace(min_val, max_val, 7).round(3)

    for i, feature in enumerate(sorted_features):
        if df[feature].min() == 0:
            _df = (
                df.copy()
                .replace(0, 1e-10)
                .sort_values(feature)
            )
        else:
            _df = (
                df.copy()
                .sort_values(feature)
            )

        # compute quantiles
        quantiles = np.percentile(_df[feature], [2.5, 10, 25, 75, 90, 97.5])
        quantiles = quantiles.tolist()

        sns.kdeplot(
            data=_df,
            x=feature,
            ax=axes[i],
            fill=True,
            log_scale=log_scale,
            color='grey',
            edgecolor='lightgrey',
        )

        # fill space between each pair of quantiles
        for j in range(len(quantiles) - 1):
            axes[i].fill_between(
                [quantiles[j],  # lower bound
                 quantiles[j + 1]],  # upper bound
                0,  # max y=0
                axes[i].get_ylim()[1] * .2,
                color=colors[j]
            )
        # mean and median values as reference
        mean = _df[feature].mean()
        median = _df[feature].median()
        axes[i].scatter([mean], [axes[i].get_ylim()[1] * .1], color='black', s=10)
        axes[i].scatter([median], [axes[i].get_ylim()[1] * .1], color='darkred', s=10, marker='^')

        # display word on left
        axes[i].text(
            max_val * 1.1, axes[i].get_ylim()[1] * .5,
            feature.title().replace('_', ' ')[0:-1],
            ha='right',
            fontsize=10,
            fontweight='bold',
            color=darkgrey
        )

        axes[i].set_yscale('linear')  # Manter escala linear no eixo Y
        axes[i].set_ylabel('')  # Remover o rótulo do eixo Y

        # x axis scale for last and first ax
        if i == 0:
            axes[i].text(
                values[0], axes[i].get_ylim()[1] * 2.3,
                f'KDE plot for selected features',
                ha='left',
                fontsize=11,
                fontweight='bold',
                color='black'
            )
            axes[i].text(
                values[-1], axes[i].get_ylim()[1] * 1.9,
                f'Variables',
                ha='right',
                fontsize=11,
                fontweight='bold',
                color='darkred'
            )
        if i == len(sorted_features) - 1:
            for value in values:
                axes[i].text(
                    value, axes[i].get_ylim()[1] * -.3,
                    f'{value}',
                    ha='center',
                    fontsize=9
                )

            axes[i].text(
                values[3], axes[i].get_ylim()[1] * -.9,
                f'{name} {xlabel_aid}',
                ha='center',
                fontsize=10,
                fontweight='bold',
                color=darkgrey
            )

        axes[i].set_axis_off()

    # details
    text = """
    Horizontal bars represent percentiles [2.5%, 10%, 25%, 75%, 90%, 97.5%].
    Black dots represent the mean value for each feature.
    Red triangles represent the median value for each feature.
    Y scale not shared across features.
    """
    fig.text(
        0, -0.1,
        text,
        ha='left',
        fontsize=8,
    )
    # Ajusta o mesmo limite log para todos os eixos X
    if transform != 'zscore':  # Garante que o limite inferior em log não resulte em erro
        axes[-1].set_xlim(min_val, max_val)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def plot_outliers(df: pd.DataFrame, features: list[str], transform: str = None):
    # Step 1: Select numerical data for outlier detection
    if transform == 'zscore':
        _df = df.loc[:, features].copy()
        for feature in features:
            _df[feature] = scipy.stats.zscore(_df[feature])
        label_aid = ' (Z-score)'
    elif transform == 'log':
        _df = df.loc[:, features].copy()
        for feature in features:
            _df[feature] = np.log(_df[feature].replace(0, 1e-10))
        label_aid = ' (Log scale)'
    elif transform == 'minmax':
        _df = df.loc[:, features].copy()
        for feature in features:
            _df[feature] = sklearn.preprocessing.minmax_scale(_df[feature])
        label_aid = ' (MinMax scale)'
    else:
        _df = df.loc[:, features].copy()
        label_aid = ''

    # Step 2: Apply Isolation Forest for outlier detection
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_predictions = iso_forest.fit_predict(_df.dropna(axis=1, how='any'))

    # Add outlier flag to the dataset (-1 for outlier, 1 for inlier)
    _df['Outlier'] = outlier_predictions

    # Step 3: Visualize Outliers using a pair plot
    plt.figure(figsize=(10, 8))
    fig = sns.pairplot(_df, hue='Outlier', palette={1: 'lightgrey', -1: 'red'},
                       diag_kind='kde', corner=True, markers='x')
    plt.suptitle(
        f'Outlier Detection with Isolation Forest (Red = Outlier) {label_aid}',
        fontsize=16,
        y=1.02
    )
    st.pyplot(fig)
    conn = st.session_state.get("conn")
    _df['Diagnosis'] = conn.execute("SELECT Diagnosis FROM breast_cancer").fetchdf()
    st.write(
        f'Correlation between Outlier Flag and Diagnosis: '
        f'{round(_df["Outlier"].corr(_df["Diagnosis"].map({"B": 0, "M": 1})), 2)}')

def plot_pairplot(df: pd.DataFrame, features: list[str], hue: str = None, transform: str = None):
    features = features + [hue] if hue else features

    if transform == 'zscore':
        _df = df.loc[:, features].copy()
        for feature in features:
            _df[feature] = scipy.stats.gzscore(_df[feature]) if feature != hue else _df[feature]
        label_aid = ' (Z-score)'
    elif transform == 'log':
        _df = df.loc[:, features].copy()
        for feature in features:
            _df[feature] = np.log(_df[feature].replace(0, 1e-10)) if feature != hue else _df[feature]
        label_aid = ' (Log scale)'
    elif transform == 'minmax':
        _df = df.loc[:, features].copy()
        for feature in features:
            _df[feature] = sklearn.preprocessing.minmax_scale(_df[feature]) if feature != hue else _df[feature]
        label_aid = ' (MinMax scale)'
    else:
        _df = df.loc[:, features].copy()
        label_aid = ''

    pairplot_fig = sns.pairplot(_df[features], hue=hue, diag_kind='kde', markers='x', palette='Set1', hue_order=['M', 'B'])
    st.pyplot(pairplot_fig.figure)

def plot_plotly_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None):
    """
    Gera um scatter plot usando Plotly Express.
    """
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                     title=f"Scatter Plot: {x_col} vs {y_col}",
                     template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig, use_container_width=True)

def plot_mean_roc_curves(roc_data: dict):
    """
    Plot mean ROC curves for multiple models.

    Args:
        roc_data (dict): Dictionary with model names and ROC curve data.
    """
    plt.figure(figsize=(10, 6))

    for model_name, data in roc_data.items():
        plt.plot(data["fpr"], data["tpr"], label=f"{model_name} (AUC = {data['auc']:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curves")
    plt.legend()
    st.pyplot(plt)
