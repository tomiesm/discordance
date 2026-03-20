"""
Regression model wrappers with unified interface.

Supports Ridge, XGBoost, and MLP for morphology -> expression prediction.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class Regressor(ABC):
    """Base class for all regressors."""

    def __init__(self, pca_components: int = 256):
        """
        Args:
            pca_components: Number of PCA components to use
        """
        self.pca_components = pca_components
        self.pca = None

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            Y_val: Optional[np.ndarray] = None):
        """
        Fit the regression model.

        Args:
            X: Training features (n_samples, n_features)
            Y: Training targets (n_samples, n_targets)
            X_val: Validation features (optional)
            Y_val: Validation targets (optional)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples, n_targets)
        """
        pass

    def fit_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA on training data and transform.

        Args:
            X: Training features (n_samples, n_features)

        Returns:
            Transformed features (n_samples, pca_components)
        """
        self.pca = PCA(n_components=self.pca_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        return X_pca

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted PCA.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Transformed features (n_samples, pca_components)
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_pca first.")
        return self.pca.transform(X)

    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        """Load model from disk."""
        return joblib.load(path)


class FixedAlphaRidgeRegressor(Regressor):
    """
    Ridge regression with fixed alpha = 100 / (n_pca_components * n_targets).
    Per HEST benchmark protocol (v3).
    """

    def __init__(self, pca_components: int = 256):
        super().__init__(pca_components)
        self.model = None

    def fit(self, X: np.ndarray, Y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            Y_val: Optional[np.ndarray] = None):
        from sklearn.linear_model import Ridge

        X_pca = self.fit_pca(X)
        alpha = 100.0 / (self.pca_components * Y.shape[1])
        self.model = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr')
        self.model.fit(X_pca, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X_pca = self.transform_pca(X)
        return self.model.predict(X_pca)


class RidgeRegressor(Regressor):
    """
    Ridge regression with cross-validation for alpha selection.
    Matches HEST-Benchmark protocol.
    """

    def __init__(self, pca_components: int = 256,
                 alphas: list = None):
        """
        Args:
            pca_components: Number of PCA components
            alphas: List of alpha values to try in CV
        """
        super().__init__(pca_components)

        if alphas is None:
            alphas = np.logspace(-3, 3, 20).tolist()

        self.alphas = alphas
        self.model = None

    def fit(self, X: np.ndarray, Y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            Y_val: Optional[np.ndarray] = None):
        """
        Fit Ridge regression with multi-output support.

        Args:
            X: Training features (n_samples, n_features)
            Y: Training targets (n_samples, n_targets)
            X_val: Not used (Ridge uses internal CV)
            Y_val: Not used (Ridge uses internal CV)
        """
        # Apply PCA
        X_pca = self.fit_pca(X)

        # Fit multi-output Ridge with cross-validation
        self.model = RidgeCV(alphas=self.alphas, cv=5)
        self.model.fit(X_pca, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict gene expression.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples, n_genes)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_pca = self.transform_pca(X)
        return self.model.predict(X_pca)


class XGBoostRegressor(Regressor):
    """
    XGBoost with separate model per gene.
    Captures nonlinear relationships.
    """

    def __init__(self, pca_components: int = 256,
                 n_estimators: int = 500,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 early_stopping_rounds: int = 20,
                 n_jobs: int = -1):
        """
        Args:
            pca_components: Number of PCA components
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Fraction of samples per tree
            colsample_bytree: Fraction of features per tree
            early_stopping_rounds: Early stopping patience
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        super().__init__(pca_components)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs

        self.models = []  # One model per gene

    def fit(self, X: np.ndarray, Y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            Y_val: Optional[np.ndarray] = None):
        """
        Fit separate XGBoost model for each gene.

        Args:
            X: Training features (n_samples, n_features)
            Y: Training targets (n_samples, n_genes)
            X_val: Validation features (optional, for early stopping)
            Y_val: Validation targets (optional, for early stopping)
        """
        # Apply PCA
        X_pca = self.fit_pca(X)

        n_genes = Y.shape[1]
        self.models = []

        # Prepare validation data if provided
        if X_val is not None and Y_val is not None:
            X_val_pca = self.transform_pca(X_val)
            eval_set = [(X_val_pca, None)]  # Will be set per gene
            use_early_stopping = True
        else:
            eval_set = None
            use_early_stopping = False

        # Train one model per gene
        for i in range(n_genes):
            y_train = Y[:, i]

            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                n_jobs=self.n_jobs,
                random_state=42,
                verbosity=0
            )

            if use_early_stopping:
                y_val = Y_val[:, i]
                model.fit(
                    X_pca, y_train,
                    eval_set=[(X_val_pca, y_val)],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False
                )
            else:
                model.fit(X_pca, y_train)

            self.models.append(model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict gene expression using all gene-specific models.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples, n_genes)
        """
        if not self.models:
            raise ValueError("Models not fitted. Call fit() first.")

        X_pca = self.transform_pca(X)

        # Predict with each model
        predictions = np.zeros((X.shape[0], len(self.models)), dtype=np.float32)
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_pca)

        return predictions


class MLPRegressor(Regressor):
    """
    Multi-layer perceptron for multi-output regression.
    PyTorch implementation with early stopping.
    """

    def __init__(self, pca_components: int = 256,
                 hidden_dims: list = None,
                 dropout: float = 0.2,
                 lr: float = 0.001,
                 batch_size: int = 256,
                 max_epochs: int = 100,
                 patience: int = 10,
                 weight_decay: float = 0.0001,
                 device: str = "cuda"):
        """
        Args:
            pca_components: Number of PCA components
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            lr: Learning rate
            batch_size: Training batch size
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            weight_decay: L2 regularization
            device: Device to train on
        """
        super().__init__(pca_components)

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = None
        self.input_dim = pca_components
        self.output_dim = None

    def _build_model(self, output_dim: int) -> nn.Module:
        """
        Build MLP architecture.

        Args:
            output_dim: Number of output genes

        Returns:
            PyTorch model
        """
        layers = []

        # Input layer
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, Y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            Y_val: Optional[np.ndarray] = None):
        """
        Fit MLP with early stopping.

        Args:
            X: Training features (n_samples, n_features)
            Y: Training targets (n_samples, n_genes)
            X_val: Validation features (required for early stopping)
            Y_val: Validation targets (required for early stopping)
        """
        # Apply PCA
        X_pca = self.fit_pca(X)

        self.output_dim = Y.shape[1]

        # Build model
        self.model = self._build_model(self.output_dim).to(self.device)

        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_pca).float(),
            torch.from_numpy(Y).float()
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        # Validation data
        if X_val is not None and Y_val is not None:
            X_val_pca = self.transform_pca(X_val)
            X_val_tensor = torch.from_numpy(X_val_pca).float().to(self.device)
            Y_val_tensor = torch.from_numpy(Y_val).float().to(self.device)
            use_validation = True
        else:
            use_validation = False

        # Optimizer and loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_Y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)

            train_loss /= len(train_dataset)

            # Validation
            if use_validation:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, Y_val_tensor).item()

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    # Restore best model
                    self.model.load_state_dict(best_state)
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict gene expression.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples, n_genes)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_pca = self.transform_pca(X)
        X_tensor = torch.from_numpy(X_pca).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy()


def get_regressor(regressor_config: dict) -> Regressor:
    """
    Factory function to create regressor from config.

    Args:
        regressor_config: Regressor configuration dict

    Returns:
        Regressor instance

    Raises:
        ValueError: If regressor type is not recognized
    """
    regressor_type = regressor_config['type'].lower()

    if regressor_type == 'ridge_fixed':
        return FixedAlphaRidgeRegressor(
            pca_components=regressor_config.get('pca_components', 256),
        )

    elif regressor_type == 'ridge':
        return RidgeRegressor(
            pca_components=regressor_config.get('pca_components', 256),
            alphas=regressor_config.get('alphas', None)
        )

    elif regressor_type == 'xgboost':
        return XGBoostRegressor(
            pca_components=regressor_config.get('pca_components', 256),
            n_estimators=regressor_config.get('n_estimators', 500),
            max_depth=regressor_config.get('max_depth', 6),
            learning_rate=regressor_config.get('learning_rate', 0.1),
            subsample=regressor_config.get('subsample', 0.8),
            colsample_bytree=regressor_config.get('colsample_bytree', 0.8),
            early_stopping_rounds=regressor_config.get('early_stopping_rounds', 20),
            n_jobs=regressor_config.get('n_jobs', -1)
        )

    elif regressor_type == 'mlp':
        return MLPRegressor(
            pca_components=regressor_config.get('pca_components', 256),
            hidden_dims=regressor_config.get('hidden_dims', [512, 256]),
            dropout=regressor_config.get('dropout', 0.2),
            lr=regressor_config.get('lr', 0.001),
            batch_size=regressor_config.get('batch_size', 256),
            max_epochs=regressor_config.get('max_epochs', 100),
            patience=regressor_config.get('patience', 10),
            weight_decay=regressor_config.get('weight_decay', 0.0001)
        )

    else:
        raise ValueError(
            f"Unknown regressor type: {regressor_type}. "
            f"Available: ridge, xgboost, mlp"
        )
