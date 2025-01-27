import duckdb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest


def load_data() -> tuple:
    """
    Load the Breast Cancer Wisconsin dataset, initialize a DuckDB connection, and register the data as a table.

    Returns:
        tuple: (df, conn, variable_info)
            - conn (duckdb.DuckDBPyConnection): DuckDB connection.
            - variable_info (str): Metadata for the dataset.
    """
    from ucimlrepo import fetch_ucirepo

    # Load dataset from UCI repository
    dataset = fetch_ucirepo(id=17)
    df = (
        dataset.data.original
        .assign(Diagnosis=lambda x: x["Diagnosis"].astype("category"))
    )

    # Create DuckDB in-memory connection and register table
    conn = duckdb.connect(database=":memory:")
    conn.execute("CREATE TABLE breast_cancer AS SELECT * FROM df")

    # Extract metadata
    variable_info = (
        dataset.metadata["additional_info"]["variable_info"]
        .replace("\r", "")
        .replace("\n\n", "", 1)
    )

    return df, conn, variable_info


def get_features_by_suffix(conn, suffix: int):
    """
    Retrieve feature columns based on their suffix (e.g., mean values, max values).

    Args:
        conn (duckdb.DuckDBPyConnection): DuckDB connection.
        suffix (int): Suffix identifier (1, 2, or 3).

    Returns:
        list: List of column names with the specified suffix.
    """
    query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'breast_cancer' AND column_name LIKE '%{suffix}'
    """
    result = conn.execute(query).fetchall()
    return [row[0] for row in result]


def apply_pca(feature_data, variance_threshold: float = 0.95, fitted_pca=None):
    """
    Apply PCA to feature groups and return transformed features.

    Args:
        feature_data (pd.DataFrame): Input DataFrame.
        variance_threshold (float): Variance retention threshold for PCA.

    Returns:
        dict: Transformed PCA features for each group.
    """
    scaler = MinMaxScaler()

    # Scale and apply PCA
    scaled_data = scaler.fit_transform(feature_data)
    pca = PCA(n_components=variance_threshold)
    transformed_data = pca.fit_transform(scaled_data)

    return pca, transformed_data


def detect_outliers(feature_data, contamination: float = 0.05):
    """
    Detect outliers using Isolation Forest and add a flag to the dataset.

    Args:
        feature_data (pd.DataFrame): Input DataFrame.
        contamination (float): Contamination parameter for Isolation Forest.

    Returns:
        duckdb.DuckDBPyRelation: Queryable DuckDB relation with outlier flags.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_flags = iso_forest.fit_predict(feature_data)

    return iso_forest, outlier_flags
