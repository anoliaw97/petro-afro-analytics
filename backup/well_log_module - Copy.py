# well_log_module.py
import streamlit as st
from well_log_analyzer import WellLogAnalyzer, create_example_model

def show():
    """Main function to display the Well Log Analysis module"""
    st.header("Well Log Analysis")
    st.markdown("### PETRO-AFRO Advanced Analytic Framework")
    
    # Create example model
    create_example_model()
    
    # Initialize session state to track app state
    if 'well_log_analyzer' not in st.session_state:
        st.session_state.well_log_analyzer = WellLogAnalyzer()
    
    analyzer = st.session_state.well_log_analyzer
    
    # Application flow options
    flow_options = ["Train New Model", "Use Existing Model for Prediction"]
    selected_flow = st.radio("Select Workflow", flow_options)
    
    if selected_flow == "Train New Model":
        # Upload Dataset section
        st.header("Data Upload and Preprocessing")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload Well Log CSV for Training", type="csv", key="well_log_file")
        
        with col2:
            # Add a clear session button
            if st.button("Reset Session", key="reset_session_button"):
                # Clear critical session states
                for key in list(st.session_state.keys()):
                    if key not in ['epochs', 'batch_size', 'learning_rate']:
                        st.session_state.pop(key, None)
                st.success("Session reset successfully. Please upload a new dataset.")
                st.experimental_rerun()
        
        if uploaded_file is not None:
            # Check if this is a new file upload
            if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.last_uploaded_file = uploaded_file.name
                if analyzer.load_data(uploaded_file):
                    analyzer.explain_features()
        
        # Only show additional sections if data is loaded
        if analyzer.df is not None:
            # Data Analysis section
            st.header("Data Analysis")
            
            # Simple button instead of expander
            if st.button("Perform Data Analysis", key="perform_analysis"):
                analyzer.analyze_data()
            
            # Model Training section
            st.header("Model Training")
            analyzer.prepare_model()
            
            # Prediction section
            st.header("Prediction")
            
            pred_tab1, pred_tab2 = st.tabs(["Single Value Prediction", "Batch Prediction from CSV"])
            
            with pred_tab1:
                analyzer.predict_new_data()
                
            with pred_tab2:
                st.subheader("Upload New Dataset for Prediction")
                st.write("Upload a new CSV file containing the same features as your training data to make predictions on multiple samples.")
                
                prediction_file = st.file_uploader("Upload CSV for Prediction", type="csv", key="prediction_file")
                
                if prediction_file is not None and analyzer.model is not None:
                    if st.button("Make Batch Predictions", key="batch_predict_button"):
                        analyzer.predict_from_csv(prediction_file)
                elif analyzer.model is None:
                    st.warning("Please train or load a model first before making batch predictions.")
            
            # Report section
            st.header("Session Report")
            if st.button("Generate Report", key="generate_report_main"):
                analyzer.save_report()
    
    else:
        # Direct Prediction Flow - No Training Needed
        st.header("Model Selection")
        
        # Get available models
        saved_models = analyzer.get_saved_models()
        
        if not saved_models:
            st.warning("No saved models found. Please train a model first or check the models directory.")
        else:
            st.write("#### Select a pre-trained model")
            selected_model = st.selectbox("Model", saved_models, key="direct_model_select")
            
            if st.button("Load Selected Model", key="load_direct_model"):
                success = analyzer.load_saved_model(selected_model)
                if success:
                    st.success(f"Model '{selected_model}' loaded successfully!")
            
            if analyzer.model is not None:
                st.header("Make Predictions")
                
                pred_tab1, pred_tab2 = st.tabs(["Single Value Prediction", "Batch Prediction from CSV"])
                
                with pred_tab1:
                    analyzer.predict_new_data()
                    
                with pred_tab2:
                    st.subheader("Upload Dataset for Prediction")
                    st.write("Upload a CSV file containing the same features as the model was trained on.")
                    
                    prediction_file = st.file_uploader("Upload CSV for Prediction", type="csv", key="direct_prediction_file")
                    
                    if prediction_file is not None:
                        if st.button("Make Batch Predictions", key="direct_batch_predict"):
                            analyzer.predict_from_csv(prediction_file)

if __name__ == "__main__":
    show()