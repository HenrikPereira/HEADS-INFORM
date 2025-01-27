from tools import *
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score


def prepare_data(conn):
    """
    Prepares data for modeling:
    - Maps 'Diagnosis' to numeric.
    - Excludes 'ID' column.
    """
    return conn.execute("""
        SELECT
            CASE WHEN Diagnosis = 'M' THEN 1 ELSE 0 END AS Diagnosis,
            *
        EXCLUDE ID, Diagnosis
        FROM breast_cancer
    """).fetchdf()


def train_and_evaluate(model, conn, splits, feature_engineering=False):
    """
    Train and evaluate a model using repeated stratified K-fold cross-validation.
    Feature engineering is applied within each fold if enabled.
    """
    aucs, f1_scores, precision_scores, recall_scores = [], [], [], []
    all_fpr = np.linspace(0, 1, 100)
    mean_tprs = []

    plt.figure(figsize=(6, 4))

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Query train and test data for this fold
        train_data = conn.execute(f"""
            SELECT * FROM breast_cancer
            WHERE ROWID IN ({', '.join(map(str, train_idx))})
        """).fetchdf()

        test_data = conn.execute(f"""
            SELECT * FROM breast_cancer
            WHERE ROWID IN ({', '.join(map(str, test_idx))})
        """).fetchdf()

        X_train = train_data.drop(columns=["Diagnosis", "ID"])
        y_train = train_data["Diagnosis"].map({'B': 0, 'M': 1,})

        X_test = test_data.drop(columns=["Diagnosis", "ID"])
        y_test = test_data["Diagnosis"].map({'B': 0, 'M': 1,})

        # Feature engineering within each fold
        if feature_engineering:
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
                "Texture Cluster": texture_features,
            }

            scaler = MinMaxScaler()
            pca_data_train = pd.DataFrame()
            pca_data_test = pd.DataFrame()

            # Apply PCA to each feature cluster
            for cluster_name, features in feature_groups.items():
                # Fit scaler and PCA on X_train
                cluster_scaled_train = scaler.fit_transform(X_train[features])
                pca = PCA(n_components=0.95)
                cluster_pca_train = pca.fit_transform(cluster_scaled_train)

                # Apply the same scaler and PCA to X_test
                cluster_scaled_test = scaler.transform(X_test[features])
                cluster_pca_test = pca.transform(cluster_scaled_test)

                for i in range(cluster_pca_train.shape[1]):
                    pca_data_train[f"{cluster_name}_PCA_{i + 1}"] = cluster_pca_train[:, i]
                    pca_data_test[f"{cluster_name}_PCA_{i + 1}"] = cluster_pca_test[:, i]

            # Add Isolation Forest outlier feature
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            outlier_predictions_train = iso_forest.fit_predict(X_train)
            outlier_predictions_test = iso_forest.predict(X_test)
            pca_data_train['Outlier'] = outlier_predictions_train
            pca_data_test['Outlier'] = outlier_predictions_test

            # Replace X_train and X_test with engineered features
            pca_data_train['Diagnosis'] = y_train.reset_index(drop=True)
            X_train = pca_data_train.drop(columns=['Diagnosis'])
            X_test = pca_data_test

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        aucs.append(auc)
        f1_scores.append(f1_score(y_test, model.predict(X_test)))
        precision_scores.append(precision_score(y_test, model.predict(X_test)))
        recall_scores.append(recall_score(y_test, model.predict(X_test)))

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        interp_tpr = np.interp(all_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        mean_tprs.append(interp_tpr)

        # Plot ROC curve for the current fold
        plt.plot(fpr, tpr, alpha=0.2, color="lightgrey")

    # Aggregate metrics
    mean_tpr = np.mean(mean_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    auc_std = np.std(aucs)
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)

    # Plot mean ROC curve
    plt.plot(all_fpr, mean_tpr, label=f"Mean ROC (AUC = {mean_auc:.2f})")
    plt.fill_between(all_fpr, mean_tpr - auc_std, mean_tpr + auc_std, alpha=0.2, color="blue")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    sns.despine()

    # Display plots and metrics
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plt, clear_figure=True)
    with col2:
        st.write(f"**Mean AUC**: {mean_auc:.2f} ({ci_lower:.2f}, {ci_upper:.2f}) 95% CI")
        st.write(f"**Mean F1 Score**: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
        st.write(f"**Mean Precision**: {np.mean(precision_scores):.2f} ± {np.std(precision_scores):.2f}")
        st.write(f"**Mean Recall**: {np.mean(recall_scores):.2f} ± {np.std(recall_scores):.2f}")

    return {
        "Mean AUC": mean_auc,
        "AUC CI Lower": ci_lower,
        "AUC CI Upper": ci_upper,
        "Mean F1": np.mean(f1_scores),
        "Mean Precision": np.mean(precision_scores),
        "Mean Recall": np.mean(recall_scores),
        "Mean TPR": mean_tpr,
        "FPR": all_fpr,
    }


def raw_models(conn):
    st.subheader("Original Features Modeling")
    st.write("""
           This section evaluates machine learning models using the dataset's **original features** without additional 
           feature engineering.

           Models included:
           - Logistic Regression
           - Random Forest
           - XGBoost

           Each model is validated using **repeated stratified k-fold cross-validation**, ensuring consistent 
           class distribution in training and test sets. Metrics like AUC, F1, Precision, and Recall are calculated 
           to assess performance.

           You can view the **ROC curve for each cross-validation fold** and the **mean ROC curve** for each model.
           """)
    data = prepare_data(conn)
    X = data.drop(columns=["Diagnosis"])
    y = data["Diagnosis"]

    skf = RepeatedStratifiedKFold(n_splits=2, n_repeats=25, random_state=42)
    splits = list(skf.split(X, y))

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    for model_name, model in models.items():
        st.subheader(model_name)
        result = train_and_evaluate(model, conn, splits, feature_engineering=False)
        if "raw_results" not in st.session_state:
            st.session_state["raw_results"] = {}
        st.session_state["raw_results"][model_name] = result


def eng_models(conn):
    st.subheader("Engineered Features Modeling")
    st.write("""
            This section evaluates machine learning models on a dataset enhanced with **feature engineering**, 
            which includes:
            - **PCA-transformed clusters** for `Size`, `Shape`, and `Texture` feature groups.
            - An **Outlier Feature** generated using Isolation Forest to detect anomalies in the data.

            Feature engineering is applied **within each cross-validation fold**, ensuring no data leakage.

            Models included:
            - Logistic Regression
            - Random Forest
            - XGBoost

            Similar to the previous section, cross-validation ensures a robust evaluation, with metrics and ROC curves 
            available for detailed analysis.
            """)
    data = prepare_data(conn)
    X = data.drop(columns=["Diagnosis"])
    y = data["Diagnosis"]

    skf = RepeatedStratifiedKFold(n_splits=2, n_repeats=25, random_state=42)
    splits = list(skf.split(X, y))

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    for model_name, model in models.items():
        st.subheader(model_name)
        result = train_and_evaluate(model, conn, splits, feature_engineering=True)
        if "eng_results" not in st.session_state:
            st.session_state["eng_results"] = {}
        st.session_state["eng_results"][model_name] = result

def metrics():
    st.subheader("Global Metrics")
    st.write("""
        Compare all models using aggregated metrics and visualize their mean ROC curves.

        Features include:
        - A **metrics table** summarizing mean AUC, F1, Precision, and Recall scores for all models.
        - A **combined ROC plot** showing the mean ROC curve for every model, allowing visual comparison of performance.

        Use this section to identify the best-performing model and assess the impact of feature engineering.
        """)
    # Retrieve saved results from session state
    raw_results = st.session_state.get('raw_results', {})
    eng_results = st.session_state.get('eng_results', {})

    # Combine all results
    all_results = []
    for model_name, result in raw_results.items():
        all_results.append({
            "Model": f"Raw {model_name}",
            "Mean AUC": f"{result['Mean AUC']} ({result['AUC CI Lower']:.3f}, {result['AUC CI Upper']:.3f})",
            "Mean F1": result["Mean F1"],
            "Mean Precision": result["Mean Precision"],
            "Mean Recall": result["Mean Recall"],
        })

    for model_name, result in eng_results.items():
        all_results.append({
            "Model": f"Engineered {model_name}",
            "Mean AUC": f"{result['Mean AUC']} ({result['AUC CI Lower']:.3f}, {result['AUC CI Upper']:.3f})",
            "Mean F1": result["Mean F1"],
            "Mean Precision": result["Mean Precision"],
            "Mean Recall": result["Mean Recall"],
        })

    # Convert results to DataFrame
    metrics_df = pd.DataFrame(all_results)

    col1, col2 = st.columns(2, gap='small', vertical_alignment='center')
    with col1:
        # Display metrics table
        st.subheader("Metrics Summary")
        st.dataframe(metrics_df, hide_index=True)
    with col2:
        # Plot ROC curves for all models
        st.subheader("Mean ROC Curves for All Models")
        plt.figure(figsize=(10, 6))
        for model_name, result in raw_results.items():
            plt.plot(result["FPR"], result["Mean TPR"], label=f"Raw {model_name} (AUC = {result['Mean AUC']:.2f})")

        for model_name, result in eng_results.items():
            plt.plot(result["FPR"], result["Mean TPR"], label=f"Engineered {model_name} (AUC = {result['Mean AUC']:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)  # Random guess line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Mean ROC Curves")
        plt.legend(loc="lower right")
        sns.despine()
        st.pyplot(plt)

# Modeling Workflow
def modeling(df):
    st.title("Modeling")
    tabs = st.tabs(["Original Features Modeling", "Engineered Features Modeling", "Global Metrics"])
    with tabs[0]:
        raw_models(df)
    with tabs[1]:
        eng_models(df)
    with tabs[2]:
        metrics()


# Run the modeling
conn = st.session_state.get("conn")
modeling(conn)
