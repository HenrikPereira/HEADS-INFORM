from tools import *

conn = st.session_state.get("conn")

def nd_feat():
    st.subheader("New & Derived Features")

    st.markdown('''
        To improve further modeling, we can do feature engineering, by creating new and derived features.

        To do that, there are several techniques that can be applied:
        ''')
    with st.expander('#### **Isolation Forest Outlier Feature**', expanded=True):
        st.markdown('''
            - **What**: Include outlier scores or flags from the Isolation Forest as an additional feature.
            - **Why**: Highlights anomalous points that may be diagnostically relevant.

            This technique is the same as the one used in the EDA section:
            ''')
        st.page_link(
            label="*Go to Isolation Forest (EDA)*",
            page="http://localhost:8501/exploratory#search-for-outliers",
            icon='🔗',
        )
    with st.expander('**Interaction Terms (Optional)**'):
        st.markdown('''
            - **What**: Create interaction terms between key features or PCA components.
            - **Examples**:
              - Ratios like `radius1 / perimeter1` or `area1 / perimeter1` for compactness.
              - Products like `PCA1_Size * PCA1_Texture` to explore joint effects of different clusters.
            - **Why**: Captures relationships that aren’t apparent in individual features.
            - **Outcome**: Test if these interactions improve predictive power.
            ''')
    with st.expander('**Non-Linear Clustering with DBSCAN (Optional)**'):
        st.markdown('''
            - **What**: Apply DBSCAN to PCA-transformed data to detect non-linear clusters or structure in the data.
            - **Why**: Reveals potential patterns that linear PCA and correlation-based clustering might miss.
            - **Outcome**: Add cluster memberships as categorical features.
            ''')
    with st.expander('**Polynomial or Log Transformations (Optional)**'):
        st.markdown('''
            - **What**: Apply polynomial terms or log transformations to specific features.
            - **Why**: Addresses potential non-linear relationships (e.g., quadratic trends) and adjusts for skewness in features.
            - **Outcome**: Test for improved feature representation.
            ''')

    st.markdown('''
        Simplifying our approach we'll go for the first technique, 
        alongside with PCA dimensionality reduction (see next tab for more details).
        ''')


def dim_red(conn):
    st.subheader("Dimensionality Reduction with PCA")
    st.write("The goal of dimensionality reduction is to reduce the number of features to a manageable size while "
             "preserving some explainability, either of the context of data, and also the original features "
             "variance (information retention).")
    st.write("PCA is a linear dimensionality reduction method that uses Singular Value Decomposition (SVD), so if we "
             "find non linear relationships between features, PCA might not be able to capture them. The alternative "
             "would be the use of tSNE, which is a nonlinear dimensionality reduction method.")

    # Define feature groups
    size_features = [f"{item}{suffix}" for item in ['radius', 'perimeter', 'area'] for suffix in range(1, 4)]
    shape_features = [f"{item}{suffix}" for item in ['concavity', 'concave_points'] for suffix in range(1, 4)]
    texture_features = [
        f"{item}{suffix}"
        for item in ['texture', 'smoothness', 'compactness', 'symmetry', 'fractal_dimension']
        for suffix in range(1, 4)
    ]

    feature_groups = {
        "Size Cluster": size_features,
        "Shape Cluster": shape_features,
        "Texture Cluster": texture_features
    }

    pca_results = {}
    pca_summary = {}
    explained_variances = {}

    # Apply PCA to each feature group
    for group_name, features in feature_groups.items():
        st.write(f"#### Applying PCA for {group_name}")

        # Select feature data
        # Scale features using MinMaxScaler
        scaler = MinMaxScaler()
        query = f"SELECT {', '.join(features)} FROM breast_cancer"
        feature_data = scaler.fit_transform(conn.execute(query).fetchdf())

        # Initialize PCA with all components
        pca = PCA()
        pca.fit(feature_data)

        # Calculate cumulative explained variance
        cumulative_variance = pca.explained_variance_ratio_.cumsum()

        # Find the number of components where the cumulative variance >= 0.95
        n_components = next(
            i for i, total_variance in enumerate(cumulative_variance, start=1) if total_variance >= 0.95)

        # Refit PCA with the selected number of components
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(feature_data)

        # Store results
        pca_results[group_name] = transformed
        explained_variances[group_name] = pca.explained_variance_ratio_

        explained_variance_percentage = cumulative_variance[n_components - 1] * 100
        remark = (
            "High information retention." if explained_variance_percentage >= 95 else
            "Moderate information retention." if explained_variance_percentage >= 85 else
            "Low information retention. Consider evaluating features or thresholds."
        )

        pca_summary[group_name] = {
            "Original Features": len(features),
            "PCA Components": n_components,
            "Explained Variance": f'{explained_variance_percentage:.2f} %',
            "Remark": remark
        }

        # Display explained variance ratio
        col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
        with col1:
            # Plot cumulative explained variance
            fig = px.line(
                x=range(1, len(cumulative_variance) + 1),
                y=cumulative_variance,
                title=f"Cumulative Explained Variance: {group_name}",
                labels={"x": "Number of Components", "y": "Cumulative Explained Variance"},
                range_x=[0.9, len(cumulative_variance) + 1],
                range_y=[0, 1.1],
            ).update_layout(
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(1, len(cumulative_variance) + 1))
                )

            )
            st.plotly_chart(fig)
        with col2:
            st.markdown(f"##### **{group_name} Summary**")
            st.dataframe(pd.DataFrame([pca_summary[group_name]]), hide_index=True)

        st.divider()

    # Summary of Findings section
    st.markdown("### Summary of Findings")
    st.markdown("The dimensionality reduction process uses PCA to combine original features into fewer, "
                "informative components. Below is an overview of dimensionality reduction per feature group/cluster"
                " including the explained variance and number of PCA components selected for optimal results.")
    st.write("The information retention with selected components for each feature group/cluster"
             " is high.")

    # Summary table
    st.dataframe(pd.DataFrame.from_dict(pca_summary, orient='index')[
                     ["Original Features", "PCA Components", "Explained Variance", "Remark"]])


def feature_engineering(conn):
    st.title("Feature Engineering")
    tabs = st.tabs(["New & Derived Features", "Dimensionality Reduction",])
    with tabs[0]:
        nd_feat()
    with tabs[1]:
        dim_red(conn)

feature_engineering(conn)
