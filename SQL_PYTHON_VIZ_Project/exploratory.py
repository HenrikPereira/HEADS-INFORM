from tools import *

conn = st.session_state.get("conn")

# Retrieve feature sets
mean_val_cols = get_features_by_suffix(conn, suffix=1)
max_val_cols = get_features_by_suffix(conn, suffix=2)
se_val_cols = get_features_by_suffix(conn, suffix=3)

def descriptive(conn):
    st.write("#### Descriptive statistics for category ***Diagnosis***")

    # Descriptive stats for Diagnosis
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        diagnosis_stats = conn.execute("""
            SELECT 
                COUNT(*) AS Total_Records,
                SUM(CASE WHEN Diagnosis = 'M' THEN 1 ELSE 0 END) AS Malignant_Count,
                SUM(CASE WHEN Diagnosis = 'B' THEN 1 ELSE 0 END) AS Benign_Count
            FROM breast_cancer
        """).fetchdf()
        st.write(diagnosis_stats.T)

    with col2:
        st.write("Class Imbalance:")
        st.write("Malignant cases (M): ", diagnosis_stats["Malignant_Count"].iloc[0])
        st.write("Benign cases (B): ", diagnosis_stats["Benign_Count"].iloc[0])
        st.info("Note: The dataset is imbalanced, which may affect classification models.")

    with col3:
        counts = conn.execute("""
            SELECT Diagnosis, COUNT(*) AS Count 
            FROM breast_cancer GROUP BY Diagnosis
        """).fetchdf()
        st.bar_chart(counts.set_index("Diagnosis")["Count"], horizontal=True)
        st.write('This unbalance of the classes may be a problem for some machine learning algorithms, '
                 'especially in classification. This has to be taken into consideration when designing models.')

    if st.checkbox("Show random sample of data"):
        sample_data = conn.execute("SELECT * FROM breast_cancer ORDER BY RANDOM() LIMIT 5").fetchdf()
        st.dataframe(sample_data, hide_index=True)

    st.divider()
    st.write("#### Descriptive Statistics for Numerical Features")
    st.write("##### ***Mean Values***")
    mean_stats = conn.execute(f"SELECT {', '.join(mean_val_cols)} FROM breast_cancer").fetchdf().describe()
    st.dataframe(mean_stats)

    st.write("##### ***Max Values***")
    max_stats = conn.execute(f"SELECT {', '.join(max_val_cols)} FROM breast_cancer").fetchdf().describe()
    st.dataframe(max_stats)

    st.write("##### ***Standard Errors***")
    se_stats = conn.execute(f"SELECT {', '.join(se_val_cols)} FROM breast_cancer").fetchdf().describe()
    st.dataframe(se_stats)

    st.write('We can see, with no surprise, that each feature has a different distribution, different range, '
             'and no apparent missing values.')

def univariate(conn):
    st.write("#### Analysis by Set of Features")
    col1, col2 = st.columns(2, gap="small")
    with col1:
        feat_set = st.selectbox("Pick a Variable Set", ["Mean Values", "Max Values", "Standard Error"])
    with col2:
        radio = st.radio(
            "Transformation & Scale",
            ["No scaling", "Log scale", "Z-score", "Min-max"],
            horizontal=True
        )

    feat_dict = {
        "Mean Values": mean_val_cols,
        "Max Values": max_val_cols,
        "Standard Error": se_val_cols,
    }
    radio_dict = {
        "No scaling": None,
        "Log scale": "log",
        "Z-score": "zscore",
        "Min-max": "minmax"
    }

    selected_features = feat_dict.get(feat_set)
    radio = radio_dict.get(radio)
    feature_data = conn.execute(f"SELECT {', '.join(selected_features)} FROM breast_cancer").fetchdf()

    st.write("Each set of features can be represented by a ridgeline plot:")
    plot_ridgeline_plot_seaborn_alt(feature_data, selected_features, feat_set, radio)

    st.write("For further analysis of the ditribution of each feature, we can use histograms, "
             "and QQ plots to check visually for normality:")
    col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
    with col1:
        hist_col = st.selectbox("Select a Variable for Histogram & QQ Plot", selected_features)
        st.info(
            '''
            Visually, every feature have distinct distributions of its data.

            Either from the Ridgeline, Histograms, and QQ plots, every feature don't follow a normal distribution.

            When rescaling with a Log or Zscore, most variables follow a normal distribution. 
            '''
        )
    with col2:
        if st.button("Generate Plot"):
            plot_histogram_qq(feature_data, hist_col, radio)

    st.write("##### Search for outliers")

    plot_outliers(feature_data, selected_features, radio)

    st.info(
        '''
        The majority of points fall within expected bounds (grey), forming dense clusters.

        Outliers (red) deviate significantly from these clusters, often being extreme values in one or both dimensions.

        The removal of these outliers can help improve the performance of machine learning algorithms.
        '''
    )

def multivariate(conn):
    st.write("#### Analysis by Pair of Features")
    col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
    with col1:
        feat_set = st.selectbox(
            "Select a Feature Set", ["Mean Values", "Max Values", "Standard Error"], key="Multi_Feature_Set"
        )
    with col2:
        radio = st.radio(
            "Transformation & Scale",
            ["No scaling", "Log scale", "Z-score", "Min-max"],
            horizontal=True,
            key="Multi_Transform"
        )

    feat_dict = {
        "Mean Values": mean_val_cols,
        "Max Values": max_val_cols,
        "Standard Error": se_val_cols,
    }

    radio_dict = {
        "No scaling": None,
        "Log scale": "log",
        "Z-score": "zscore",
        "Min-max": "minmax"
    }

    selected_features = feat_dict.get(feat_set)
    radio = radio_dict.get(radio)
    pairplot_data = conn.execute(f"SELECT {', '.join(selected_features)}, Diagnosis FROM breast_cancer").fetchdf()

    # Pairplot visualization
    st.write("Visualize pairwise relationships in the selected feature set:")
    plot_pairplot(pairplot_data, selected_features, hue="Diagnosis", transform=radio)

    with st.expander("Detailed analysis by pair of features"):
        col1, col2, col3 = st.columns(3, gap="small", vertical_alignment="center")
        with col1:
            x_axis = st.selectbox("X-axis", selected_features, key="eda_x_axis")
        with col2:
            y_axis = st.selectbox("Y-axis", selected_features, key="eda_y_axis")
        with col3:
            color_axis = st.checkbox("Hue by Diagnosis", value=True, key="eda_color_axis")

        plot_plotly_scatter(pairplot_data, x_axis, y_axis, 'Diagnosis' if color_axis else None)

    st.info(
        '''
        A clear distinction between diagnostic cases is visible in the pairplot for most of feature pairs.

        Due to some shapes in some pair, there is also a hint of some multicollinearity.

        Most pairs have linear relationship, specially if a log transformation is applied.
        '''
    )

def correlations(conn):
    st.write("#### Hierarchical Clustering Correlation Heatmap")
    radio = st.radio("Transformation & Scale", ["No scaling", "Min-max"], horizontal=True)

    numeric_data = conn.execute("""
        SELECT 
            CASE WHEN Diagnosis = 'M' THEN 1 ELSE 0 END AS Diagnosis_Num,
            radius1, radius2, radius3, 
            texture1, texture2, texture3,
            smoothness1, smoothness2, smoothness3,
            compactness1, compactness2, compactness3,
            concavity1, concavity2, concavity3,
            concave_points1, concave_points2, concave_points3,
            symmetry1, symmetry2, symmetry3,
            fractal_dimension1, fractal_dimension2, fractal_dimension3,
            perimeter1, perimeter2, perimeter3,
            area1, area2, area3
        FROM breast_cancer
    """).fetchdf()

    if radio == "Min-max":
        numeric_data = numeric_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    correlation_matrix = numeric_data.corr()

    # Hierarchical clustering heatmap
    plt.figure(figsize=(16, 12))
    fig = sns.clustermap(
        correlation_matrix,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        annot=False,
        figsize=(14, 10),
        method='ward',
        cbar=True,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
    )
    st.pyplot(fig)

    st.info(
        '''
        We can see that there are some hierarchical correlations between multiple features

        Taking into account our target variable 'Diagnosis', we can make some inferences:
        '''
    )
    st.markdown(
        '''
        ##### **Cluster 1: Size and Perimeter Features**
        - **Key Features**: `radius1`, `radius2`, `radius3`, `perimeter1`, `perimeter2`, `perimeter3`, `area1`, `area2`, `area3`
        - **Interpretation**: 
          - These features measure the size and boundary of cells.
          - Strong correlations among these variables suggest they collectively describe similar properties of the cell, such as overall size or shape.
          - Example: Larger radii and perimeters typically result in larger areas, leading to their high correlation.

        ---

        ##### **Cluster 2: Concavity and Concave Points**
        - **Key Features**: `concavity1`, `concavity2`, `concavity3`, `concave_points1`, `concave_points2`, `concave_points3`
        - **Interpretation**:
          - These features describe the concavity of the cell's shape (indentations or irregularities).
          - High inter-correlation implies they jointly represent the irregularity of cell boundaries, which could be crucial in distinguishing benign from malignant cells.
          - Strong correlations with `Diagnosis` indicate their importance in detecting malignancy.

        ---

        ##### **Cluster 3: Smoothness and Compactness**
        - **Key Features**: `smoothness1`, `smoothness2`, `smoothness3`, `compactness1`, `compactness2`, `compactness3`
        - **Interpretation**:
          - These features describe cell surface uniformity (smoothness) and tightness (compactness).
          - Compactness might relate to cell density, while smoothness measures evenness, making this cluster a key descriptor of cell structure.

        ---

        ##### **Cluster 4: Symmetry and Fractal Dimension**
        - **Key Features**: `symmetry1`, `symmetry2`, `symmetry3`, `fractal_dimension1`, `fractal_dimension2`, `fractal_dimension3`
        - **Interpretation**:
          - These features relate to the geometric and structural balance of cells.
          - Symmetry might correlate with benign cases, while irregular fractal dimensions could indicate malignancy.

        ---

        ##### **Cluster Importance**
        - **Clusters 1 and 2** are likely the most diagnostic, given their high correlations with `Diagnosis`. These include size, perimeter, concavity, and concave points.
        - **Clusters 3 and 4** might provide additional supporting information but show relatively weaker direct correlations with the target variable.

        - **Interpretation**:
           - Intuitively we can see 3 patterns in the analysed features:
              - Size related -> `radius`, `perimeter`, `area`
              - Shape related -> `concavity`, `concave_points`
              - Texture related -> `texture`, `smoothness`, `compactness`, `symmetry`, `fractal_dimension`
        ---
        '''
    )

def eda(conn):
    st.title("Exploratory Data Analysis (EDA)")

    # Tabs for different analyses
    tabs = st.tabs([
        "General Description",
        "Univariate Analysis",
        "Multivariate Analysis",
        "Correlations",
    ])

    with tabs[0]:
        descriptive(conn)

    with tabs[1]:
        univariate(conn)

    with tabs[2]:
        multivariate(conn)

    with tabs[3]:
        correlations(conn)

eda(conn)
