from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import typer

from housing_price_neural_net.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class HousePriceNet(nn.Module):
    def __init__(self, input_size: int, task_type: str = "classification"):
        super().__init__()
        self.task_type = task_type
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3 if task_type == "classification" else 1)
        )

    def forward(self, x):
        return self.network(x)


def oversample_minority_classes(X: np.ndarray, y: np.ndarray, oversampling_ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes by duplicating samples.

    Args:
        X: Feature matrix
        y: Target labels
        oversampling_ratio: Ratio of minority class samples to majority class samples.
                          If 1.0, all classes will have the same number of samples.
                          If 0.5, minority classes will have half the samples of majority class.

    Returns:
        tuple: (oversampled_X, oversampled_y)
    """
    unique, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    target_count = int(max_count * oversampling_ratio)

    X_oversampled = []
    y_oversampled = []

    for cls in unique:
        cls_indices = np.where(y == cls)[0]
        cls_count = len(cls_indices)

        if cls_count < target_count:
            n_duplicates = target_count // cls_count
            remainder = target_count % cls_count

            for idx in cls_indices:
                X_oversampled.append(X[idx])
                y_oversampled.append(y[idx])

                for _ in range(n_duplicates - 1):
                    X_oversampled.append(X[idx])
                    y_oversampled.append(y[idx])

            if remainder > 0:
                random_indices = np.random.choice(cls_indices, remainder, replace=False)
                for idx in random_indices:
                    X_oversampled.append(X[idx])
                    y_oversampled.append(y[idx])
        else:
            random_indices = np.random.choice(cls_indices, target_count, replace=False)
            for idx in random_indices:
                X_oversampled.append(X[idx])
                y_oversampled.append(y[idx])

    return np.array(X_oversampled), np.array(y_oversampled)


def oversample_minority_classes_smote(X: np.ndarray, y: np.ndarray, oversampling_ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes using SMOTE algorithm.

    Args:
        X: Feature matrix
        y: Target labels
        oversampling_ratio: Ratio of minority class samples to majority class samples.
                          If 1.0, all classes will have the same number of samples.
                          If 0.5, minority classes will have half the samples of majority class.

    Returns:
        tuple: (oversampled_X, oversampled_y)
    """
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42, sampling_strategy=oversampling_ratio)
    X_oversampled, y_oversampled = smote.fit_resample(X, y)

    return X_oversampled, y_oversampled


def load_data(data_path: Path, task_type: str = "classification"):
    """Load and preprocess data.

    Args:
        data_path: Path to the data file
        task_type: Either "classification" or "regression"
    """
    logger.info("Loading data...")
    df = pd.read_csv(data_path)

    features = df.drop(columns=["price"])
    labels = df["price"]

    X = features.values.astype(float)
    y = labels.values

    if task_type == "classification":
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, y_train = oversample_minority_classes(X_train, y_train)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    return X_train, X_val, y_train, y_val


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    task_type: str = "classification",
) -> tuple[list, list]:
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            if task_type == "regression":
                outputs = outputs.squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                if task_type == "regression":
                    outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


def calculate_adjusted_weights(y: np.ndarray, scaling_factor=0.75) -> torch.Tensor:
    _, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    weights = weights ** scaling_factor
    weights = weights / weights.sum()
    return torch.FloatTensor(weights)


@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "processed_train_data.csv",
    model_path: Path = MODELS_DIR / "model.pth",
    val_data_path: Path = MODELS_DIR / "val_data.npz",
    batch_size: int = 32,
    num_epochs: int = 10000,
    learning_rate: float = 0.001,
    oversampling_ratio: float = 1.0,
    use_smote: bool = False,
    task_type: str = "classification",
):
    """Train a neural network model for house price prediction.

    Args:
        data_path: Path to CSV file with features and price_class column
        model_path: Path to save the trained model
        val_data_path: Path to save validation data
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        oversampling_ratio: Ratio of minority class samples to majority class samples
        use_smote: Whether to use SMOTE instead of simple oversampling
    """
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    X_train, X_val, y_train, y_val = load_data(data_path, task_type)

    np.savez(val_data_path, X_val=X_val, y_val=y_val)
    logger.info(f"Validation data saved to {val_data_path}")

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_size = X_train.shape[1]
    model = HousePriceNet(input_size, task_type).to(device)

    class_weights = calculate_adjusted_weights(y_train)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device, task_type
    )

    torch.save(model.state_dict(), model_path)
    logger.success(f"Model saved to {model_path}")


if __name__ == "__main__":
    app()
