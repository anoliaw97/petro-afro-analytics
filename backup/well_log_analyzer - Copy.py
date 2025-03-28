import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import importlib.util
import time
import io
import base64
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Feature descriptions dictionary
FEATURE_DESCRIPTIONS = {
    "Depth": "Depth of well with measurements taken every 0.5m",
    "RxoRt": "Ratio of Shallow and deep resistivity in the well",
    "RLL3": "Laterlog 3 resistivity data",
    "SP": "Spontaneous potential data",
    "RILD": "Deep Induction resistivity data",
    "MN": "Resistivity wide array",
    "MI": "Resistivity Intermediate Array",
    "MCAL": "Normal caliper, which shows the size of well",
    "DCAL": "Differential Caliper",
    "RHOB": "Bulk density data",
    "RHOC": "Corrected bulk density",
    "DPOR": "Density porosity",
    "CNLS": "Compensated neutron log",
    "GR": "Gamma ray log"
}

class WellLogAnalyzer:
    def __init__(self):
        # Initialize class attributes directly
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = None
        self.target_column = None
        self.correlation_matrix = None
        self.training_results = None
        self.prediction_result = None
        self.dataset_info = None
        
        # Storage for restored values from session state
        if 'df' in st.session_state:
            self.df = st.session_state.df
        if 'model' in st.session_state:
            self.model = st.session_state.model
        if 'feature_columns' in st.session_state:
            self.feature_columns = st.session_state.feature_columns
        if 'target_column' in st.session_state:
            self.target_column = st.session_state.target_column
        if 'scaler_X' in st.session_state:
            self.scaler_X = st.session_state.scaler_X
        if 'scaler_y' in st.session_state:
            self.scaler_y = st.session_state.scaler_y
        if 'correlation_matrix' in st.session_state:
            self.correlation_matrix = st.session_state.correlation_matrix
        if 'training_results' in st.session_state:
            self.training_results = st.session_state.training_results
        if 'prediction_result' in st.session_state:
            self.prediction_result = st.session_state.prediction_result
        if 'dataset_info' in st.session_state:
            self.dataset_info = st.session_state.dataset_info
                
        # Models directory
        self.models_dir = "models/well_log"
        self.ensure_directory(self.models_dir)
    
    def ensure_directory(self, directory):
        """Utility function to create directory if it doesn't exist"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            return True
        return False
    
    def reset_session(self):
        """Reset session state when loading a new dataset"""
        # Reset analysis and model-related attributes
        self.X = None
        self.y = None
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = None
        self.target_column = None
        self.correlation_matrix = None
        self.training_results = None
        self.prediction_result = None
        
        # Update session state
        st.session_state.X = self.X
        st.session_state.y = self.y
        st.session_state.model = self.model
        st.session_state.scaler_X = self.scaler_X
        st.session_state.scaler_y = self.scaler_y
        st.session_state.feature_columns = self.feature_columns
        st.session_state.target_column = self.target_column
        st.session_state.correlation_matrix = self.correlation_matrix
        st.session_state.training_results = self.training_results
        st.session_state.prediction_result = self.prediction_result

    def load_data(self, uploaded_file):
        """Load and validate CSV file"""
        try:
            # Reset session for new dataset
            self.reset_session()
            
            # Read the CSV file
            self.df = pd.read_csv(uploaded_file)
            st.session_state.df = self.df
            
            # Check for numeric columns
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            
            # Handle missing values
            missing_data = self.df[numeric_columns].isnull().sum()
            
            # Store dataset info for reporting
            self.dataset_info = {
                'filename': uploaded_file.name,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'shape': self.df.shape,
                'columns': list(self.df.columns),
                'numeric_columns': list(numeric_columns),
                'missing_values': missing_data.sum()
            }
            st.session_state.dataset_info = self.dataset_info
            
            st.success("Data loaded successfully!")
            
            # Display data info
            st.write("### Dataset Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Dataset Shape:", self.df.shape)
                st.write("Numeric Columns:", list(numeric_columns))
            
            with col2:
                st.write("Missing Values:")
                st.dataframe(missing_data[missing_data > 0])
            
            # Option to handle missing values
            if missing_data.sum() > 0:
                fill_method = st.selectbox(
                    "How would you like to handle missing values?", 
                    ["Drop rows with missing values", "Fill with mean"],
                    key="missing_values_method"
                )
                
                if fill_method == "Drop rows with missing values":
                    self.df.dropna(inplace=True)
                    self.dataset_info['missing_values_handling'] = 'dropped'
                    st.session_state.df = self.df
                    st.session_state.dataset_info = self.dataset_info
                else:
                    self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
                    self.dataset_info['missing_values_handling'] = 'filled_with_mean'
                    st.session_state.df = self.df
                    st.session_state.dataset_info = self.dataset_info
            
            # Store correlation matrix
            self.correlation_matrix = self.df.select_dtypes(include=[np.number]).corr()
            st.session_state.correlation_matrix = self.correlation_matrix
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False

    def explain_features(self):
        """Display explanations for each feature"""
        st.write("### Feature Explanations")
        
        # Create dataframe for feature descriptions
        feature_info = []
        for col in self.df.columns:
            description = FEATURE_DESCRIPTIONS.get(col, "No description available")
            feature_info.append({"Feature": col, "Description": description})
        
        # Display as dataframe
        st.dataframe(pd.DataFrame(feature_info))

    def analyze_data(self):
        """Comprehensive data analysis"""
        st.write("### Data Analysis")
        
        # Correlation Heatmap
        st.write("#### Correlation Heatmap")
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
        st.pyplot(plt)
        plt.close()
        
        # Descriptive Statistics
        st.write("#### Descriptive Statistics")
        st.dataframe(self.df.describe())

    def get_available_models(self):
        """Get list of saved models"""
        save_dir = os.path.join(self.models_dir, "saved")
        if not os.path.exists(save_dir):
            return []
            
        models = [f[:-3] for f in os.listdir(save_dir) if f.endswith('.h5')]
        return models

    def get_available_custom_models(self):
        """Scan the models directory for Python files containing model classes"""
        available_models = ["Default Neural Network"]
        
        # Check if models directory exists
        models_dir = "models/well_log"
        if not os.path.exists(models_dir):
            return available_models
            
        # Scan for Python files
        for file in os.listdir(models_dir):
            if file.endswith('.py'):
                model_name = file[:-3]  # Remove .py extension
                available_models.append(model_name)
                
        return available_models

    def load_model_from_file(self, model_name):
        """Load model class from Python file"""
        if model_name == "Default Neural Network":
            return None
            
        try:
            # Construct path to model file
            model_path = os.path.join("models/well_log", f"{model_name}.py")
            
            # Load module
            spec = importlib.util.spec_from_file_location(model_name, model_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            
            # Return model class (assuming the main class has the same name as the file)
            return getattr(model_module, model_name)
        except Exception as e:
            st.error(f"Failed to load model {model_name}: {str(e)}")
            return None

    def visualize_data_split(self, test_size, validation_size):
        """Create a visual representation of the data split"""
        # Calculate actual percentages
        training_pct = 100 - test_size - validation_size*(100-test_size)/100
        validation_pct = validation_size*(100-test_size)/100
        test_pct = test_size
        
        # Create figure
        plt.figure(figsize=(10, 1.5))
        
        # Create a horizontal stacked bar
        left = 0
        plt.barh(0, training_pct, left=left, height=0.5, color='#4CAF50', alpha=0.8)
        left += training_pct
        plt.barh(0, validation_pct, left=left, height=0.5, color='#2196F3', alpha=0.8)
        left += validation_pct
        plt.barh(0, test_pct, left=left, height=0.5, color='#FFC107', alpha=0.8)
        
        # Add percentage labels
        plt.text(training_pct/2, 0, f"Training\n{training_pct:.1f}%", 
                ha='center', va='center', color='white', fontweight='bold')
        plt.text(training_pct + validation_pct/2, 0, f"Validation\n{validation_pct:.1f}%", 
                ha='center', va='center', color='white', fontweight='bold')
        plt.text(training_pct + validation_pct + test_pct/2, 0, f"Test\n{test_pct:.1f}%", 
                ha='center', va='center', color='white', fontweight='bold')
        
        # Remove axes and labels
        plt.axis('off')
        plt.tight_layout()
        
        return plt

    def prepare_model(self):
        """Prepare model training interface"""
        st.write("### Model Training")
        
        # Select target and feature columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        target_column = st.selectbox(
            "Select Target Column for Prediction", 
            numeric_columns,
            key="target_column_select"
        )
        
        # Select feature columns (exclude target)
        feature_columns = [col for col in numeric_columns if col != target_column]
        
        # Initialize training parameters
        epochs_val = 100
        batch_size_val = 32
        learning_rate_val = 0.001
        
        # Get values from session state if they exist
        if 'epochs' in st.session_state:
            epochs_val = st.session_state.epochs
        if 'batch_size' in st.session_state:
            batch_size_val = st.session_state.batch_size
        if 'learning_rate' in st.session_state:
            learning_rate_val = st.session_state.learning_rate
        
        # Model selection dropdown
        available_models = self.get_available_custom_models()
        selected_model = st.selectbox("Select Model", available_models, key="model_select")
        
        # Training parameters
        st.write("#### Training Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=epochs_val, key="epochs_input")
            st.session_state.epochs = epochs
            
        with col2:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=512, value=batch_size_val, key="batch_size_input")
            st.session_state.batch_size = batch_size
            
        with col3:
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=learning_rate_val, format="%.4f", key="lr_input")
            st.session_state.learning_rate = learning_rate
        
        # Validation Split slider
        st.write("#### Data Split Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Split (% of data for testing)", 
                               min_value=10, max_value=40, value=20, key="test_split")
            test_split = test_size / 100  # Convert percentage to fraction
        
        with col2:
            validation_size = st.slider("Validation Split (% of training data for validation)", 
                                    min_value=5, max_value=30, value=20, key="validation_split")
            validation_split = validation_size / 100  # Convert percentage to fraction
        
        st.info(f"Your data will be split as follows: {100-test_size-validation_size*(100-test_size)/100:.1f}% training, "
                f"{validation_size*(100-test_size)/100:.1f}% validation, {test_size}% testing")
        
        # Display the split visualization
        split_viz = self.visualize_data_split(test_size, validation_size)
        st.pyplot(split_viz)
        plt.close()
        
        # Early Stopping options - SIMPLIFIED VERSION
        st.write("#### Early Stopping Options")

        # Just use the widgets without trying to sync with session state
        early_stopping_enabled = st.checkbox("Enable Early Stopping", value=True, key="es_enabled")

        patience = 10  # Default value
        if early_stopping_enabled:
            patience = st.slider("Patience (epochs without improvement)", 
                               min_value=1, max_value=50, value=10, key="es_patience")

        if early_stopping_enabled:
            st.info(f"Early stopping will halt training if validation loss doesn't improve for {patience} consecutive epochs.")
        
        # Model name for saving
        model_save_name = st.text_input("Model Save Name", value=f"{selected_model}_for_{target_column}", key="model_name_input")
        
        # Check if we have training results to display from a previous run
        if self.training_results is not None:
            st.success("Model has been trained. See results below.")
            self._display_training_results(self.training_results)
        
        # Create and train model button
        if st.button("Train Model", key="train_model_button"):
            # Prepare data
            X = self.df[feature_columns]
            y = self.df[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42
            )
            
            # Scale the data
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_test_scaled = self.scaler_X.transform(X_test)
            
            y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
            
            # Create model
            if selected_model == "Default Neural Network":
                model = self._create_default_neural_network(X_train_scaled.shape[1], learning_rate)
            else:
                # Load custom model class
                ModelClass = self.load_model_from_file(selected_model)
                if ModelClass is None:
                    return
                    
                # Initialize and compile model
                model = ModelClass(input_shape=(X_train_scaled.shape[1],))
                model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            
            # Prepare callbacks
            callbacks = []
            early_stopping_epoch = None
            
            # Add Early Stopping if enabled
            if early_stopping_enabled:
                early_stopping = EarlyStopping(
                    monitor='val_loss', 
                    patience=patience, 
                    restore_best_weights=True,
                    verbose=1  # This will print when early stopping is triggered
                )
                callbacks.append(early_stopping)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Custom callback for progress updates
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, analyzer):
                    super().__init__()
                    self.best_epoch = 0
                    self.best_val_loss = float('inf')
                    self.stopped_epoch = None
                    self.analyzer = analyzer
                
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Training Progress: {int(progress * 100)}% (Epoch {epoch+1}/{epochs})")
                    
                    # Track best epoch
                    if logs.get('val_loss', float('inf')) < self.best_val_loss:
                        self.best_val_loss = logs.get('val_loss')
                        self.best_epoch = epoch + 1
                    
                    time.sleep(0.1)  # Small delay to show progress
                
                def on_train_end(self, logs=None):
                    if hasattr(self.model, 'stopped_epoch') and self.model.stopped_epoch > 0:
                        self.stopped_epoch = self.model.stopped_epoch + 1
                        status_text.text(f"Training stopped early at epoch {self.stopped_epoch}. " +
                                       f"Best performance was at epoch {self.best_epoch}.")
            
            # Create and add our custom callback
            progress_callback = ProgressCallback(self)
            callbacks.append(progress_callback)
            
            # Train model
            history = model.fit(
                X_train_scaled, 
                y_train_scaled,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Check if training stopped early
            actual_epochs = len(history.history['loss'])
            early_stopping_info = None
            
            if early_stopping_enabled and actual_epochs < epochs:
                if progress_callback.stopped_epoch:
                    early_stopping_info = {
                        'stopped_epoch': progress_callback.stopped_epoch,
                        'best_epoch': progress_callback.best_epoch
                    }
                    
                    # Update progress text
                    status_text.text(f"Training stopped early at epoch {early_stopping_info['stopped_epoch']}. " +
                                   f"Best performance was at epoch {early_stopping_info['best_epoch']}.")
            else:
                # Update progress to completion
                progress_bar.progress(1.0)
                status_text.text("Training completed for all epochs!")
            
            # Evaluate model
            y_pred_scaled = model.predict(X_test_scaled).flatten()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            self.training_results = {
                'history': history.history,
                'y_test': y_test,
                'y_pred': y_pred,
                'mse': mse,
                'r2': r2,
                'model_type': selected_model,
                'target_column': target_column,
                'feature_columns': feature_columns,
                'training_parameters': {
                    'epochs': epochs,
                    'actual_epochs': actual_epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'test_split': test_split,
                    'validation_split': validation_split,
                    'early_stopping_enabled': early_stopping_enabled,
                    'early_stopping_patience': patience if early_stopping_enabled else None,
                    'early_stopping_info': early_stopping_info,
                    'train_test_split': 0.2
                },
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.training_results = self.training_results
            
            # Display results
            self._display_training_results(self.training_results)
            
            # Store model for later use
            self.model = model
            self.feature_columns = feature_columns
            self.target_column = target_column
            
            # Update session state
            st.session_state.model = self.model
            st.session_state.feature_columns = self.feature_columns
            st.session_state.target_column = self.target_column
            st.session_state.scaler_X = self.scaler_X
            st.session_state.scaler_y = self.scaler_y
            
            # Save model
            self.save_model(model_save_name, model, feature_columns, target_column)
    
    def _create_default_neural_network(self, input_shape, learning_rate):
        """Helper method to create default neural network"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        return model
    
    def _display_training_results(self, results):
        """Helper method to display training results"""
        # Unpack results
        history = results['history']
        y_test = results['y_test']
        y_pred = results['y_pred']
        mse = results['mse']
        r2 = results['r2']
        
        # Display data split information - text only, no visualization
        params = results.get('training_parameters', {})
        if 'test_split' in params and 'validation_split' in params:
            test_split = params['test_split'] * 100
            validation_split = params['validation_split'] * 100
            training_split = 100 - test_split - validation_split * (100 - test_split) / 100
            
            st.write("#### Data Split Used")
            st.text(f"Training: {training_split:.1f}%, Validation: {validation_split * (100 - test_split) / 100:.1f}%, Testing: {test_split:.1f}%")
            
        # Display early stopping information if available
        if params.get('early_stopping_enabled') and params.get('early_stopping_info'):
            info = params['early_stopping_info']
            st.info(f"Training stopped early at epoch {info['stopped_epoch']} out of {params['epochs']} total epochs. " +
                   f"Best performance was at epoch {info['best_epoch']}.")
        
        # Visualize results
        st.write("#### Model Training Results")
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('Predicted vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        st.pyplot(plt)
        plt.close()
        
        # Display metrics
        st.write("### Model Performance")
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"R-squared Score: {r2:.4f}")

    def save_model(self, model_name, model, feature_columns, target_column):
        """Save model and metadata"""
        try:
            # Create save directory if it doesn't exist
            save_dir = os.path.join(self.models_dir, "saved")
            self.ensure_directory(save_dir)
            
            # Save model
            model_path = os.path.join(save_dir, f"{model_name}.h5")
            model.save(model_path)
            
            # Save metadata (feature columns, target column, and scalers)
            metadata = {
                "feature_columns": feature_columns,
                "target_column": target_column,
                "date_saved": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Save scalers with consistent file naming
            scaler_X_path = os.path.join(save_dir, f"{model_name}_scaler_X.npy")
            scaler_y_path = os.path.join(save_dir, f"{model_name}_scaler_y.npy")
            scaler_X_var_path = os.path.join(save_dir, f"{model_name}_scaler_X_var.npy")
            scaler_y_var_path = os.path.join(save_dir, f"{model_name}_scaler_y_var.npy")
            
            # Save scaler parameters
            np.save(scaler_X_path, self.scaler_X.mean_)
            np.save(scaler_X_var_path, self.scaler_X.var_)
            np.save(scaler_y_path, self.scaler_y.mean_)
            np.save(scaler_y_var_path, self.scaler_y.var_)
            
            st.success(f"Model saved successfully as {model_name}")
        except Exception as e:
            st.error(f"Error saving model: {e}")

    def load_saved_model(self, model_name):
        """Load a saved model and its metadata"""
        try:
            save_dir = os.path.join(self.models_dir, "saved")
            
            # Load model
            model_path = os.path.join(save_dir, f"{model_name}.h5")
            self.model = load_model(model_path)
            
            # Load metadata
            metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata["feature_columns"]
            self.target_column = metadata["target_column"]
            
            # Load scalers - using os.path.join for path construction
            scaler_X_path = os.path.join(save_dir, f"{model_name}_scaler_X.npy")
            scaler_y_path = os.path.join(save_dir, f"{model_name}_scaler_y.npy")
            scaler_X_var_path = os.path.join(save_dir, f"{model_name}_scaler_X_var.npy")
            scaler_y_var_path = os.path.join(save_dir, f"{model_name}_scaler_y_var.npy")
            
            # Check if files exist before loading
            for path in [scaler_X_path, scaler_y_path, scaler_X_var_path, scaler_y_var_path]:
                if not os.path.exists(path):
                    st.error(f"Required scaler file not found: {path}")
                    return False
            
            # Initialize scalers properly
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
            # Load scaler parameters
            self.scaler_X.mean_ = np.load(scaler_X_path)
            self.scaler_X.var_ = np.load(scaler_X_var_path)
            # Properly set the scale_ attribute with np.sqrt(var_)
            self.scaler_X.scale_ = np.sqrt(self.scaler_X.var_)
            # Initialize n_features_in_ to the length of mean_ 
            self.scaler_X.n_features_in_ = len(self.scaler_X.mean_)
            
            self.scaler_y.mean_ = np.load(scaler_y_path)
            self.scaler_y.var_ = np.load(scaler_y_var_path)
            # Properly set the scale_ attribute with np.sqrt(var_)
            self.scaler_y.scale_ = np.sqrt(self.scaler_y.var_)
            # Initialize n_features_in_ to the length of mean_
            self.scaler_y.n_features_in_ = len(self.scaler_y.mean_)
            
            # Update session state
            st.session_state.model = self.model
            st.session_state.feature_columns = self.feature_columns
            st.session_state.target_column = self.target_column
            st.session_state.scaler_X = self.scaler_X
            st.session_state.scaler_y = self.scaler_y
            
            st.success(f"Model {model_name} loaded successfully")
            st.write(f"Target column: {self.target_column}")
            st.write(f"Feature columns: {self.feature_columns}")
            st.write(f"Date saved: {metadata.get('date_saved', 'Unknown')}")
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    def get_saved_models(self):
        """Get list of saved models"""
        save_dir = os.path.join(self.models_dir, "saved")
        if not os.path.exists(save_dir):
            return []
            
        models = [f[:-3] for f in os.listdir(save_dir) if f.endswith('.h5')]
        return models

    def predict_new_data(self):
        """Prediction interface"""
        st.write("### Predict New Data")
        
        # Option to load a saved model
        saved_models = self.get_saved_models()
        
        if saved_models:
            st.write("#### Load Saved Model")
            selected_model = st.selectbox("Select a saved model", [""] + saved_models, key="saved_model_select")
            
            if selected_model and st.button("Load Model", key="load_model_button"):
                self.load_saved_model(selected_model)
        
        if self.model is None:
            st.warning("Please train or load a model first!")
            return
        
        # Store prediction results
        if self.prediction_result is None:
            self.prediction_result = None
        
        st.write(f"#### Enter values for prediction of {self.target_column}")
        
        # Create input fields for features
        input_data = {}
        
        # Create columns for better layout
        cols = st.columns(3)
        for i, col in enumerate(self.feature_columns):
            with cols[i % 3]:
                # Use session state to remember values
                if f'input_{col}' not in st.session_state:
                    st.session_state[f'input_{col}'] = float(self.df[col].mean()) if self.df is not None else 0.0
                
                input_data[col] = st.number_input(
                    f"{col}", 
                    value=st.session_state[f'input_{col}'],
                    key=f"input_{col}_field",
                    help=FEATURE_DESCRIPTIONS.get(col, "No description available")
                )
                # Update session state with current value
                st.session_state[f'input_{col}'] = input_data[col]
        
        # Check if we already have prediction results to display
        if self.prediction_result is not None:
            self._display_prediction_result(self.prediction_result)
        
        if st.button("Make Prediction", key="predict_button"):
            # Prepare input data
            input_array = np.array([input_data[col] for col in self.feature_columns]).reshape(1, -1)
            
            # Scale input
            input_scaled = self.scaler_X.transform(input_array)
            
            # Predict
            prediction_scaled = self.model.predict(input_scaled).flatten()
            prediction = self.scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
            
            # Store prediction result
            self.prediction_result = {
                'prediction': prediction[0],
                'input_data': input_data,
                'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.prediction_result = self.prediction_result
            
            # Display result
            self._display_prediction_result(self.prediction_result)
    
    def _display_prediction_result(self, result):
        """Helper method to display prediction results"""
        prediction = result['prediction']
        input_data = result['input_data']
        
        # Display result
        st.success(f"Predicted {self.target_column}: {prediction:.4f}")
        
        # Optional: Add visualization of prediction
        if self.df is not None:
            # Create a scatter plot of actual vs. predicted values with the new prediction highlighted
            # Get predictions for all data points
            all_X = self.df[self.feature_columns]
            all_X_scaled = self.scaler_X.transform(all_X)
            all_predictions_scaled = self.model.predict(all_X_scaled).flatten()
            all_predictions = self.scaler_y.inverse_transform(all_predictions_scaled.reshape(-1, 1)).flatten()
            
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df[self.target_column], all_predictions, alpha=0.5, label='Dataset points')
            plt.scatter([prediction], [prediction], color='red', s=100, marker='*', 
                       label='New prediction')
            plt.plot([self.df[self.target_column].min(), self.df[self.target_column].max()], 
                    [self.df[self.target_column].min(), self.df[self.target_column].max()], 
                    'g--', label='Perfect prediction')
            plt.xlabel(f'Actual {self.target_column}')
            plt.ylabel(f'Predicted {self.target_column}')
            plt.title('Prediction Visualization')
            plt.legend()
            st.pyplot(plt)
            plt.close()

    def generate_report(self):
        """Generate a comprehensive session report"""
        # Check if we have data to report
        if self.df is None:
            st.warning("Please upload and analyze data before generating a report.")
            return
        
        # Begin generating the report
        report = []
        report.append("# Well Log Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Dataset info
        if self.dataset_info:
            report.append("## Dataset Information")
            info = self.dataset_info
            report.append(f"- **Filename**: {info.get('filename', 'Unknown')}")
            report.append(f"- **Upload Time**: {info.get('upload_time', 'Unknown')}")
            report.append(f"- **Dataset Shape**: {info.get('shape', (0, 0))}")
            report.append(f"- **Number of Features**: {len(info.get('columns', []))}")
            report.append(f"- **Missing Values**: {info.get('missing_values', 0)}")
            if 'missing_values_handling' in info:
                report.append(f"- **Missing Values Handling**: {info.get('missing_values_handling', 'None')}")
            report.append("")
        
        # Add descriptive statistics
        report.append("## Descriptive Statistics")
        if self.df is not None:
            # Convert DataFrame to Markdown
            stats_df = self.df.describe().round(3)
            stats_md = stats_df.to_markdown()
            report.append(stats_md)
            report.append("")
        
        # Training results
        if self.training_results:
            report.append("## Model Training Results")
            results = self.training_results
            report.append(f"- **Model Type**: {results.get('model_type', 'Unknown')}")
            report.append(f"- **Target Column**: {results.get('target_column', 'Unknown')}")
            report.append(f"- **Feature Columns**: {', '.join(results.get('feature_columns', []))}")
            report.append(f"- **Training Date**: {results.get('training_date', 'Unknown')}")
            report.append("")
            
            report.append("### Training Parameters")
            params = results.get('training_parameters', {})
            
            # Display validation split if available
            if 'test_split' in params and 'validation_split' in params:
                test_split = params['test_split'] * 100
                validation_split = params['validation_split'] * 100
                training_split = 100 - test_split - validation_split * (100 - test_split) / 100
                
                report.append(f"- **Data Split**: Training {training_split:.1f}%, Validation {validation_split * (100 - test_split) / 100:.1f}%, Testing {test_split:.1f}%")
            
            for param, value in params.items():
                if param not in ['early_stopping_info', 'test_split', 'validation_split']:
                    report.append(f"- **{param}**: {value}")
            
            # Add early stopping info if available
            if params.get('early_stopping_enabled') and params.get('early_stopping_info'):
                info = params['early_stopping_info']
                report.append(f"- **Early Stopping**: Training stopped at epoch {info['stopped_epoch']}")
                report.append(f"- **Best Epoch**: {info['best_epoch']}")
            report.append("")
            
            report.append("### Performance Metrics")
            report.append(f"- **Mean Squared Error**: {results.get('mse', 0):.4f}")
            report.append(f"- **R-squared Score**: {results.get('r2', 0):.4f}")
            report.append("")
        
        # Prediction results (if available)
        if self.prediction_result:
            report.append("## Latest Prediction Result")
            pred = self.prediction_result
            report.append(f"- **Prediction Time**: {pred.get('prediction_time', 'Unknown')}")
            report.append(f"- **Predicted {self.target_column}**: {pred.get('prediction', 0):.4f}")
            report.append("")
            
            report.append("### Input Values")
            for feature, value in pred.get('input_data', {}).items():
                report.append(f"- **{feature}**: {value}")
            report.append("")
        
        # Join all report sections
        report_text = "\n".join(report)
        return report_text

    def save_report(self):
        """Save the generated report"""
        report_text = self.generate_report()
        
        # Create a download link
        b64 = base64.b64encode(report_text.encode()).decode()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"well_log_report_{current_time}.md"
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Report</a>'
        
        st.markdown("### Session Report")
        st.markdown("A comprehensive report of your current session is ready for download.")
        st.markdown(href, unsafe_allow_html=True)
        
        # Preview the report
        with st.expander("Preview Report"):
            st.markdown(report_text)

def create_example_model():
    """Create an example LSTM model file if models directory exists"""
    models_dir = "models/well_log"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # Example LSTM model file
    lstm_model_path = os.path.join(models_dir, "LSTMModel.py")
    
    if not os.path.exists(lstm_model_path):
        lstm_code = """import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

class LSTMModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        return self.model
"""
        with open(lstm_model_path, 'w') as f:
            f.write(lstm_code)

# Add this method to your WellLogAnalyzer class

def predict_from_csv(self, uploaded_file):
    """Make predictions from a CSV file"""
    try:
        # Read the CSV file
        new_data = pd.read_csv(uploaded_file)
        st.write("### Dataset Overview")
        st.write(f"Number of samples: {len(new_data)}")
        
        # Check if all required features are in the CSV
        missing_features = [col for col in self.feature_columns if col not in new_data.columns]
        if missing_features:
            st.error(f"The uploaded file is missing required features: {', '.join(missing_features)}")
            return
        
        # Extract features needed for prediction
        X_new = new_data[self.feature_columns]
        
        # Add a progress bar
        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.text("Preparing data for prediction...")
        
        # Prepare data
        X_new_scaled = self.scaler_X.transform(X_new)
        
        # Update progress
        progress_bar.progress(0.3)
        progress_text.text("Making predictions...")
        
        # Make predictions
        predictions_scaled = self.model.predict(X_new_scaled)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        # Update progress
        progress_bar.progress(0.7)
        progress_text.text("Processing results...")
        
        # Add predictions to the dataframe
        new_data[f'Predicted_{self.target_column}'] = predictions
        
        # Calculate error if target column exists in the uploaded data
        if self.target_column in new_data.columns:
            new_data['Error'] = new_data[self.target_column] - new_data[f'Predicted_{self.target_column}']
            new_data['Error_Pct'] = (new_data['Error'] / new_data[self.target_column]) * 100
            
            # Calculate metrics
            mse = mean_squared_error(new_data[self.target_column], new_data[f'Predicted_{self.target_column}'])
            r2 = r2_score(new_data[self.target_column], new_data[f'Predicted_{self.target_column}'])
            st.write("#### Performance on New Dataset")
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"R-squared Score: {r2:.4f}")
            
            # Create a plot of actual vs predicted
            plt.figure(figsize=(10, 6))
            plt.scatter(new_data[self.target_column], new_data[f'Predicted_{self.target_column}'], alpha=0.5)
            plt.plot([new_data[self.target_column].min(), new_data[self.target_column].max()], 
                    [new_data[self.target_column].min(), new_data[self.target_column].max()], 
                    'r--', lw=2)
            plt.xlabel(f'Actual {self.target_column}')
            plt.ylabel(f'Predicted {self.target_column}')
            plt.title('Predicted vs Actual (New Dataset)')
            st.pyplot(plt)
            plt.close()
        
        # Complete progress
        progress_bar.progress(1.0)
        progress_text.text("Predictions complete!")
        
        # Display the results
        st.write("#### Prediction Results")
        st.dataframe(new_data)
        
        # Provide a download link for the predictions
        csv = new_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        href = f'<a href="data:file/csv;base64,{b64}" download="predictions_{current_time}.csv">Download Predictions CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")