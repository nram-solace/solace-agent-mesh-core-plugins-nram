"""Simple machine learning action for basic ML tasks."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import json

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo
from solace_ai_connector.common.log import log


class SimpleMlAction(Action):
    """Action for performing simple machine learning tasks."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "simple_ml",
                "prompt_directive": (
                    "Perform simple machine learning tasks on the loaded dataset. "
                    "This includes classification and regression with automatic preprocessing. "
                    "You can specify the target column and feature columns, or use the agent's defaults."
                ),
                "params": [
                    {
                        "name": "task_type",
                        "desc": "Type of ML task to perform",
                        "type": "string",
                        "required": True,
                        "enum": ["classification", "regression", "auto"],
                    },
                    {
                        "name": "target_column",
                        "desc": "Target column for prediction (uses agent default if not specified)",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "feature_columns",
                        "desc": "Comma-separated list of feature columns (uses all except target if not specified)",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "model_type",
                        "desc": "Type of model to use",
                        "type": "string",
                        "required": False,
                        "enum": ["random_forest", "linear", "logistic"],
                        "default": "random_forest",
                    },
                    {
                        "name": "test_size",
                        "desc": "Proportion of data to use for testing (0.1 to 0.5)",
                        "type": "float",
                        "required": False,
                        "default": 0.2,
                    },
                    {
                        "name": "random_state",
                        "desc": "Random state for reproducibility",
                        "type": "integer",
                        "required": False,
                        "default": 42,
                    },
                ],
            }
        )

    def invoke(self, request_data: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """Execute the simple ML action."""
        try:
            # Get parameters
            task_type = request_data.get("task_type", "auto")
            target_column = request_data.get("target_column")
            feature_columns_str = request_data.get("feature_columns", "")
            model_type = request_data.get("model_type", "random_forest")
            test_size = request_data.get("test_size", 0.2)
            random_state = request_data.get("random_state", 42)

            # Validate task_type
            valid_task_types = ["classification", "regression", "auto"]
            if task_type not in valid_task_types:
                return ActionResponse(
                    message=f"Invalid task_type '{task_type}'. Valid types are: {', '.join(valid_task_types)}",
                    error_info=ErrorInfo(f"Invalid task_type '{task_type}'. Valid types are: {', '.join(valid_task_types)}")
                )

            # Validate model_type
            valid_model_types = ["random_forest", "linear", "logistic"]
            if model_type not in valid_model_types:
                return ActionResponse(
                    message=f"Invalid model_type '{model_type}'. Valid types are: {', '.join(valid_model_types)}",
                    error_info=ErrorInfo(f"Invalid model_type '{model_type}'. Valid types are: {', '.join(valid_model_types)}")
                )

            # Validate test_size
            if not 0.1 <= test_size <= 0.5:
                return ActionResponse(
                    message="test_size must be between 0.1 and 0.5",
                    error_info=ErrorInfo("test_size must be between 0.1 and 0.5")
                )

            # Get the agent and its data
            agent = self.get_agent()
            
            try:
                data = agent.get_working_data().copy()
            except ValueError as e:
                return ActionResponse(
                    message=str(e),
                    error_info=ErrorInfo(str(e))
                )
            
            data_service = agent.get_data_service()

            # Determine target column
            if not target_column:
                target_column = agent.target_col
            
            if not target_column:
                return ActionResponse(
                    message="No target column specified. Please specify target_column parameter or configure it in the agent.",
                    error_info=ErrorInfo("No target column specified. Please specify target_column parameter or configure it in the agent.")
                )

            if target_column not in data.columns:
                return ActionResponse(
                    message=f"Target column '{target_column}' not found in dataset",
                    error_info=ErrorInfo(f"Target column '{target_column}' not found in dataset")
                )

            # Parse feature columns
            if feature_columns_str:
                feature_columns = [col.strip() for col in feature_columns_str.split(",") if col.strip()]
                missing_cols = [col for col in feature_columns if col not in data.columns]
                if missing_cols:
                    return ActionResponse(
                        message=f"Feature columns not found: {missing_cols}",
                        error_info=ErrorInfo(f"Feature columns not found: {missing_cols}")
                    )
            else:
                # Use all columns except target
                feature_columns = [col for col in data.columns if col != target_column]

            # Remove rows with missing target values
            data = data.dropna(subset=[target_column])
            
            if len(data) == 0:
                return ActionResponse(
                    message="No valid data rows after removing missing target values",
                    error_info=ErrorInfo("No valid data rows after removing missing target values")
                )

            # Determine task type automatically if needed
            if task_type == "auto":
                unique_values = data[target_column].nunique()
                is_numeric = pd.api.types.is_numeric_dtype(data[target_column])
                
                if is_numeric and unique_values > 10:
                    task_type = "regression"
                else:
                    task_type = "classification"
                
                log.info("ml-pandas: Auto-detected task type: %s (unique values: %d, numeric: %s)", 
                        task_type, unique_values, is_numeric)

            # Prepare features and target
            X = data[feature_columns].copy()
            y = data[target_column].copy()

            # Handle missing values in features
            X = X.fillna(X.mean(numeric_only=True))  # Fill numeric columns with mean
            X = X.fillna("Unknown")  # Fill categorical columns with "Unknown"

            # Encode categorical features
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns
            label_encoders = {}
            
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

            # Encode target if classification
            target_encoder = None
            if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Scale features for linear models
            scaler = None
            if model_type in ["linear", "logistic"]:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Train model
            model = self._create_model(model_type, task_type, random_state)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, task_type, target_encoder)

            # Get feature importance if available
            feature_importance = self._get_feature_importance(model, feature_columns)

            # Prepare results
            result = {
                "task_type": task_type,
                "model_type": model_type,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "data_info": {
                    "total_samples": len(data),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features_count": len(feature_columns),
                    "categorical_features": len(categorical_columns)
                },
                "metrics": metrics,
                "feature_importance": feature_importance,
                "preprocessing": {
                    "categorical_encoders_used": len(label_encoders),
                    "target_encoded": target_encoder is not None,
                    "features_scaled": scaler is not None
                }
            }

            # Clean result for JSON serialization
            clean_result = data_service.clean_data_for_json(result)

            # Save results
            filename = f"ml_{task_type}_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            saved_path = data_service.save_results(clean_result, filename)

            response_text = self._format_response(clean_result, saved_path)

            return ActionResponse(
                message=response_text
            )

        except Exception as e:
            log.error("ml-pandas: Error in simple ML action: %s", str(e))
            return ActionResponse(
                message=f"Failed to perform ML task: {str(e)}",
                error_info=ErrorInfo(f"Failed to perform ML task: {str(e)}")
            )

    def _create_model(self, model_type: str, task_type: str, random_state: int):
        """Create the appropriate model."""
        if task_type == "classification":
            if model_type == "random_forest":
                return RandomForestClassifier(n_estimators=100, random_state=random_state)
            elif model_type == "logistic":
                return LogisticRegression(random_state=random_state, max_iter=1000)
            else:
                return RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:  # regression
            if model_type == "random_forest":
                return RandomForestRegressor(n_estimators=100, random_state=random_state)
            elif model_type == "linear":
                return LinearRegression()
            else:
                return RandomForestRegressor(n_estimators=100, random_state=random_state)

    def _calculate_metrics(self, y_test, y_pred, task_type: str, target_encoder=None):
        """Calculate appropriate metrics."""
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get class names for report
            if target_encoder:
                target_names = target_encoder.classes_
                report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            else:
                report = classification_report(y_test, y_pred, output_dict=True)
            
            return {
                "accuracy": float(accuracy),
                "classification_report": report
            }
        else:  # regression
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            return {
                "mean_squared_error": float(mse),
                "root_mean_squared_error": float(rmse),
                "r2_score": float(r2)
            }

    def _get_feature_importance(self, model, feature_columns: List[str]):
        """Get feature importance if available."""
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_columns, model.feature_importances_))
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            return {
                "available": True,
                "features": [{"feature": feat, "importance": float(imp)} for feat, imp in sorted_importance]
            }
        else:
            return {"available": False, "reason": "Model does not provide feature importance"}

    def _format_response(self, result: Dict[str, Any], saved_path: str) -> str:
        """Format the response text."""
        response_lines = [
            f"# Machine Learning Results",
            f"**Task:** {result['task_type'].title()}",
            f"**Model:** {result['model_type'].replace('_', ' ').title()}",
            f"**Target Column:** {result['target_column']}",
            ""
        ]

        # Data info
        data_info = result["data_info"]
        response_lines.extend([
            "## Data Information",
            f"- **Total Samples:** {data_info['total_samples']:,}",
            f"- **Training Samples:** {data_info['training_samples']:,}",
            f"- **Test Samples:** {data_info['test_samples']:,}",
            f"- **Features Used:** {data_info['features_count']}",
            f"- **Categorical Features:** {data_info['categorical_features']}",
            ""
        ])

        # Metrics
        metrics = result["metrics"]
        response_lines.append("## Model Performance")
        
        if result["task_type"] == "classification":
            response_lines.extend([
                f"- **Accuracy:** {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
                ""
            ])
            
            # Add precision/recall for each class
            if "classification_report" in metrics:
                report = metrics["classification_report"]
                if "macro avg" in report:
                    macro_avg = report["macro avg"]
                    response_lines.extend([
                        "### Overall Metrics (Macro Average)",
                        f"- **Precision:** {macro_avg['precision']:.4f}",
                        f"- **Recall:** {macro_avg['recall']:.4f}",
                        f"- **F1-Score:** {macro_avg['f1-score']:.4f}",
                        ""
                    ])
        else:  # regression
            response_lines.extend([
                f"- **R² Score:** {metrics['r2_score']:.4f}",
                f"- **RMSE:** {metrics['root_mean_squared_error']:.4f}",
                f"- **MSE:** {metrics['mean_squared_error']:.4f}",
                ""
            ])

        # Feature importance
        if result["feature_importance"]["available"]:
            response_lines.extend([
                "## Top 10 Most Important Features",
                ""
            ])
            
            top_features = result["feature_importance"]["features"][:10]
            for i, feat_info in enumerate(top_features, 1):
                response_lines.append(
                    f"{i}. **{feat_info['feature']}:** {feat_info['importance']:.4f}"
                )
            response_lines.append("")

        response_lines.extend([
            "---",
            f"**Results saved to:** `{saved_path}`",
            "",
            "💡 **Available task types:** classification, regression, auto",
            "🤖 **Available models:** random_forest, linear, logistic"
        ])

        return "\n".join(response_lines)