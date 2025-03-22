from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
import typer

from housing_price_neural_net.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

CONTINUOUS_COLUMNS = [
    "YearBuilt",
    "Size(sqf)",
    "Floor",
    "N_Parkinglot(Ground)",
    "N_Parkinglot(Basement)",
    "N_manager",
    "N_elevators",
    "N_FacilitiesInApt",
    "N_FacilitiesNearBy(Total)",
    "N_SchoolNearBy(Total)",
]

CATEGORICAL_NOMINAL_COLUMNS = [
    "HallwayType",
    "HeatingType",
    "AptManageType",
    "SubwayStation",
]


CATEGORICAL_ORDINAL_COLUMNS = [
    "TimeToBusStop",
    "TimeToSubway",
]

CHEAP_THRESHOLD = 100_000
AVERAGE_THRESHOLD = 350_000


def convert_price_to_class(price: float) -> int:
    """Convert price to class label.
    
    Args:
        price: Price in dollars
        
    Returns:
        int: Class label (0: cheap, 1: average, 2: expensive)
    """
    if price <= CHEAP_THRESHOLD:
        return 0
    elif price <= AVERAGE_THRESHOLD:
        return 1
    else:
        return 2


def create_ordinal_mapping(values: pd.Series) -> dict:
    """Create mapping for ordinal values based on available data.
    
    Args:
        values: Series with values to map
        
    Returns:
        dict: Mapping from values to ordinal numbers
    """
    unique_values = sorted(values.dropna().unique())
    
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    
    return mapping


def encode_ordinal_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Encode a column as ordinal numbers and scale to [-1, 1].
    
    Args:
        df: DataFrame with the column to encode
        column: Name of the column to encode
        
    Returns:
        DataFrame with encoded column
    """
    mapping = create_ordinal_mapping(df[column])
    
    df[column] = df[column].map(mapping)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[column] = scaler.fit_transform(df[[column]])
    
    return df


def scale_continuous_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale continuous features to range [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[CONTINUOUS_COLUMNS] = scaler.fit_transform(df[CONTINUOUS_COLUMNS])
    return df


def encode_ordinal_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical ordinal features as ordinal numbers and scale to [-1, 1]."""
    for col in CATEGORICAL_ORDINAL_COLUMNS:
        df = encode_ordinal_column(df, col)
    return df


def encode_categorical_features(df: pd.DataFrame, use_one_hot: bool = True) -> pd.DataFrame:
    """Encode categorical features either as one-hot encoding or ordinal numbers.
    
    For one-hot encoding, values remain 0/1.
    For ordinal encoding, values are scaled to [-1, 1].
    """
    if use_one_hot:
        df = pd.get_dummies(df, columns=CATEGORICAL_NOMINAL_COLUMNS, prefix=CATEGORICAL_NOMINAL_COLUMNS)
    else:
        for col in CATEGORICAL_NOMINAL_COLUMNS:
            df = encode_ordinal_column(df, col)
    return df


def process_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert SalePrice to class labels and separate features from labels.
    
    Args:
        df: DataFrame with SalePrice column
        
    Returns:
        tuple: (features DataFrame, labels Series)
    """
    labels = df["SalePrice"].apply(convert_price_to_class)
    
    features = df.drop(columns=["SalePrice"])
    
    return features, labels


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "train_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    use_one_hot: bool = True,
):
    """Generate features from the dataset.
    
    Args:
        input_path: Path to input dataset
        output_path: Path to save processed features with labels
        use_one_hot: Whether to use one-hot encoding for categorical features
    """
    logger.info("Loading dataset...")
    df = pd.read_csv(input_path)
    
    logger.info("Converting prices to classes...")
    features, labels = process_labels(df)

    features["price_class"] = labels
    
    logger.info("Scaling continuous features...")
    features = scale_continuous_features(features)
    
    logger.info("Encoding time features...")
    features = encode_ordinal_time_features(features)
    
    logger.info("Encoding categorical features...")
    features = encode_categorical_features(features, use_one_hot)
    
    logger.info("Saving processed features with labels...")
    features.to_csv(output_path, index=False)
    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
