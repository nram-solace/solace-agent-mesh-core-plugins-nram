"""ML Service for handling machine learning operations."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import joblib
import pickle
from datetime import datetime
import warnings

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from solace_ai_connector.common.log import log


class MLService:
    """Service class for handling machine learning operations."""

    def __init__(self, random_state: int = 42, parallel_jobs: int = -1, 
                 max_memory_usage: int = 1024, enable_caching: bool = True):
        """Initialize the ML service.

        Args:
            random_state: Random state for reproducibility
            parallel_jobs: Number of parallel jobs for computations
            max_memory_usage: Maximum memory usage in MB
            enable_caching: Enable model and computation caching
        """
        self.random_state = random_state
        self.parallel_jobs = parallel_jobs
        self.max_memory_usage = max_memory_usage
        self.enable_caching = enable_caching
        
        # Set random state for all operations
        np.random.seed(random_state)
        
        # Initialize caches
        self._model_cache = {}
        self._data_cache = {}
        
        # Configure matplotlib for non-interactive backend
        plt.switch_backend('Agg')
        
        # Set seaborn style
        sns.set_style("whitegrid")
        
        log.info("ML Service initialized with random_state=%d, parallel_jobs=%d", 
                random_state, parallel_jobs)

    def get_regression_models(self) -> Dict[str, Any]:
        """Get available regression models.

        Returns:
            Dictionary of available regression models.
        """
        return {
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(random_state=self.random_state),
            "lasso_regression": Lasso(random_state=self.random_state),
            "random_forest": RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state, 
                n_jobs=self.parallel_jobs
            ),
            "svr": SVR(),
            "knn": KNeighborsRegressor(n_jobs=self.parallel_jobs),
            "decision_tree": DecisionTreeRegressor(random_state=self.random_state),
        }

    def get_classification_models(self) -> Dict[str, Any]:
        """Get available classification models.

        Returns:
            Dictionary of available classification models.
        """
        return {
            "logistic_regression": LogisticRegression(random_state=self.random_state),
            "random_forest": RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state, 
                n_jobs=self.parallel_jobs
            ),
            "svc": SVC(random_state=self.random_state, probability=True),
            "knn": KNeighborsClassifier(n_jobs=self.parallel_jobs),
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(random_state=self.random_state),
        }

    def get_outlier_detection_methods(self) -> Dict[str, Any]:
        """Get available outlier detection methods.

        Returns:
            Dictionary of available outlier detection methods.
        """
        return {
            "isolation_forest": IsolationForest(
                random_state=self.random_state, 
                n_jobs=self.parallel_jobs
            ),
            "local_outlier_factor": LocalOutlierFactor(
                n_jobs=self.parallel_jobs, 
                novelty=True
            ),
            "elliptic_envelope": EllipticEnvelope(random_state=self.random_state),
            "one_class_svm": OneClassSVM(),
        }

    def preprocess_data(self, df: pd.DataFrame, target_column: str, 
                       feature_columns: List[str], task_type: str = "regression") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Preprocess data for machine learning.

        Args:
            df: Input DataFrame
            target_column: Target column name
            feature_columns: List of feature column names
            task_type: Type of task (regression or classification)

        Returns:
            Tuple of (X, y, preprocessing_info)
        """
        preprocessing_info = {
            "scalers": {},
            "encoders": {},
            "feature_names": feature_columns.copy(),
            "target_name": target_column
        }

        # Handle missing values
        df_clean = df.copy()
        
        # For features, fill missing values with median for numeric, mode for categorical
        for col in feature_columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else "Unknown", inplace=True)

        # For target, remove rows with missing values
        df_clean = df_clean.dropna(subset=[target_column])

        # Prepare features
        X = df_clean[feature_columns].copy()
        y = df_clean[target_column].copy()

        # Encode categorical features
        for col in feature_columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                preprocessing_info["encoders"][col] = le

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        preprocessing_info["scalers"]["features"] = scaler

        # Handle target encoding for classification
        if task_type == "classification" and y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            preprocessing_info["encoders"]["target"] = le_target

        return X_scaled, y, preprocessing_info

    def train_regression_model(self, X: np.ndarray, y: np.ndarray, 
                              model_type: str = "random_forest", 
                              test_size: float = 0.2,
                              hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """Train a regression model.

        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of regression model
            test_size: Test set size
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary containing model and results.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Get model
        models = self.get_regression_models()
        if model_type not in models:
            raise ValueError(f"Unsupported model type: {model_type}")

        model = models[model_type]

        # Hyperparameter tuning
        if hyperparameter_tuning:
            param_grids = {
                "random_forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "ridge_regression": {
                    "alpha": [0.1, 1.0, 10.0, 100.0]
                },
                "lasso_regression": {
                    "alpha": [0.1, 1.0, 10.0, 100.0]
                }
            }
            
            if model_type in param_grids:
                grid_search = GridSearchCV(
                    model, param_grids[model_type], 
                    cv=5, scoring='r2', n_jobs=self.parallel_jobs
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=self.parallel_jobs)
        metrics["cv_r2_mean"] = cv_scores.mean()
        metrics["cv_r2_std"] = cv_scores.std()

        return {
            "model": model,
            "model_type": model_type,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "metrics": metrics,
            "feature_importance": self._get_feature_importance(model, model_type)
        }

    def train_classification_model(self, X: np.ndarray, y: np.ndarray, 
                                  model_type: str = "random_forest", 
                                  test_size: float = 0.2,
                                  hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """Train a classification model.

        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of classification model
            test_size: Test set size
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary containing model and results.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Get model
        models = self.get_classification_models()
        if model_type not in models:
            raise ValueError(f"Unsupported model type: {model_type}")

        model = models[model_type]

        # Hyperparameter tuning
        if hyperparameter_tuning:
            param_grids = {
                "random_forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "logistic_regression": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ['l1', 'l2']
                }
            }
            
            if model_type in param_grids:
                grid_search = GridSearchCV(
                    model, param_grids[model_type], 
                    cv=5, scoring='accuracy', n_jobs=self.parallel_jobs
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "train_precision": precision_score(y_train, y_pred_train, average='weighted'),
            "test_precision": precision_score(y_test, y_pred_test, average='weighted'),
            "train_recall": recall_score(y_train, y_pred_train, average='weighted'),
            "test_recall": recall_score(y_test, y_pred_test, average='weighted'),
            "train_f1": f1_score(y_train, y_pred_train, average='weighted'),
            "test_f1": f1_score(y_test, y_pred_test, average='weighted'),
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=self.parallel_jobs)
        metrics["cv_accuracy_mean"] = cv_scores.mean()
        metrics["cv_accuracy_std"] = cv_scores.std()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)

        return {
            "model": model,
            "model_type": model_type,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "metrics": metrics,
            "confusion_matrix": cm,
            "feature_importance": self._get_feature_importance(model, model_type)
        }

    def detect_outliers(self, X: np.ndarray, method: str = "isolation_forest", 
                       contamination: float = 0.1) -> Dict[str, Any]:
        """Detect outliers in the data.

        Args:
            X: Feature matrix
            method: Outlier detection method
            contamination: Expected proportion of outliers

        Returns:
            Dictionary containing outlier detection results.
        """
        methods = self.get_outlier_detection_methods()
        if method not in methods:
            raise ValueError(f"Unsupported outlier detection method: {method}")

        detector = methods[method]
        
        # Set contamination for methods that support it
        if hasattr(detector, 'contamination'):
            detector.contamination = contamination

        # Fit and predict
        if hasattr(detector, 'fit_predict'):
            outlier_labels = detector.fit_predict(X)
            # Convert to boolean (1 for outliers, -1 for inliers)
            is_outlier = outlier_labels == -1
        else:
            detector.fit(X)
            outlier_labels = detector.predict(X)
            # Convert to boolean (1 for outliers, -1 for inliers)
            is_outlier = outlier_labels == -1

        outlier_indices = np.where(is_outlier)[0]
        inlier_indices = np.where(~is_outlier)[0]

        return {
            "method": method,
            "contamination": contamination,
            "outlier_indices": outlier_indices,
            "inlier_indices": inlier_indices,
            "n_outliers": len(outlier_indices),
            "n_inliers": len(inlier_indices),
            "outlier_ratio": len(outlier_indices) / len(X),
            "outlier_labels": outlier_labels
        }

    def perform_eda(self, df: pd.DataFrame, target_column: str = None, 
                   feature_columns: List[str] = None) -> Dict[str, Any]:
        """Perform exploratory data analysis.

        Args:
            df: Input DataFrame
            target_column: Target column name (optional)
            feature_columns: List of feature column names (optional)

        Returns:
            Dictionary containing EDA results.
        """
        eda_results = {
            "basic_info": {},
            "descriptive_stats": {},
            "missing_values": {},
            "correlations": {},
            "distributions": {},
            "target_analysis": {}
        }

        # Basic information
        eda_results["basic_info"] = {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }

        # Descriptive statistics
        eda_results["descriptive_stats"] = df.describe().to_dict()

        # Missing values
        missing_data = df.isnull().sum()
        eda_results["missing_values"] = {
            "missing_counts": missing_data.to_dict(),
            "missing_percentages": (missing_data / len(df) * 100).to_dict(),
            "total_missing": missing_data.sum()
        }

        # Correlations (for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            eda_results["correlations"] = {
                "correlation_matrix": correlation_matrix.to_dict(),
                "high_correlations": self._find_high_correlations(correlation_matrix)
            }

        # Target analysis
        if target_column and target_column in df.columns:
            target_data = df[target_column]
            eda_results["target_analysis"] = {
                "unique_values": target_data.nunique(),
                "value_counts": target_data.value_counts().to_dict(),
                "is_numeric": target_data.dtype in [np.number],
                "missing_values": target_data.isnull().sum()
            }

            if target_data.dtype in [np.number]:
                eda_results["target_analysis"].update({
                    "mean": target_data.mean(),
                    "median": target_data.median(),
                    "std": target_data.std(),
                    "min": target_data.min(),
                    "max": target_data.max(),
                    "skewness": target_data.skew(),
                    "kurtosis": target_data.kurtosis()
                })

        return eda_results

    def save_model(self, model: Any, model_name: str, file_path: str) -> str:
        """Save a trained model to disk.

        Args:
            model: Trained model object
            model_name: Name of the model
            file_path: Path to save the model

        Returns:
            Path where model was saved.
        """
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(model, file_path)
            
            log.info("Model '%s' saved to %s", model_name, file_path)
            return file_path
        except Exception as e:
            raise ValueError(f"Failed to save model: {str(e)}")

    def load_model(self, file_path: str) -> Any:
        """Load a trained model from disk.

        Args:
            file_path: Path to the saved model

        Returns:
            Loaded model object.
        """
        try:
            model = joblib.load(file_path)
            log.info("Model loaded from %s", file_path)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    def _get_feature_importance(self, model: Any, model_type: str) -> Dict[str, float]:
        """Get feature importance from a trained model.

        Args:
            model: Trained model
            model_type: Type of model

        Returns:
            Dictionary of feature importance scores.
        """
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(range(len(model.feature_importances_)), model.feature_importances_))
            elif hasattr(model, 'coef_'):
                return dict(zip(range(len(model.coef_[0])), abs(model.coef_[0])))
            else:
                return {}
        except Exception:
            return {}

    def _find_high_correlations(self, correlation_matrix: pd.DataFrame, 
                               threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Find highly correlated feature pairs.

        Args:
            correlation_matrix: Correlation matrix
            threshold: Correlation threshold

        Returns:
            List of highly correlated feature pairs.
        """
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_value
                    ))
        return high_correlations 