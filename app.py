import streamlit as st
import os
import sys
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="PETRO-AFRO Analytics Framework",
    page_icon="🛢️",
    layout="wide"
)

# Navigation using query parameters and session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Check query parameters
query_params = st.experimental_get_query_params()
if 'page' in query_params:
    st.session_state.page = query_params['page'][0]

# Function to navigate to a page
def navigate_to(page_name):
    st.session_state.page = page_name
    st.experimental_set_query_params(page=page_name)

# Function to read updates from a text file
def read_updates():
    """Read updates from updates.txt file"""
    try:
        # Create the file if it doesn't exist
        if not os.path.exists("updates.txt"):
            with open("updates.txt", "w") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d')}] Welcome to PETRO-AFRO Analytics Framework!\n")
                f.write(f"[{datetime.now().strftime('%Y-%m-%d')}] Version 1.0 released with Well Log Analysis module.\n")
        
        # Read the file
        with open("updates.txt", "r") as f:
            updates = f.readlines()
        
        # Filter out empty lines and strip whitespace
        updates = [update.strip() for update in updates if update.strip()]
        
        return updates
    except Exception as e:
        return [f"Error reading updates: {str(e)}"]

# Custom CSS for enhanced menu
st.markdown("""
<style>
    /* Main titles */
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        color: #4F4F4F;
    }
    
    /* Menu cards */
    div.big-button > button {
        width: 100% !important;
        height: 180px !important;
        background-color: white !important;
        border: 2px solid #f0f0f0 !important;
        border-radius: 10px !important;
        text-align: left !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
        transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s !important;
    }
    div.big-button > button:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important;
        border-color: #4e8df5 !important;
    }
    div.big-button > button > div {
        font-size: 1.8rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .button-icon {
        font-size: 3rem !important;
        margin-bottom: 1rem !important;
    }
    .button-title {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    .stMarkdown p {
        font-size: 1.05rem !important;
    }
    
    /* Section dividers */
    .section-divider {
        margin: 40px 0 !important;
        border-top: 1px solid #eee !important;
    }
    
    /* Step list formatting */
    .steps-container ol li {
        font-size: 1.1rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Updates section */
    .updates-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .update-item {
        padding: 10px 0;
        border-bottom: 1px solid #eee;
    }
    .update-date {
        font-weight: bold;
        color: #4e8df5;
    }
    .update-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .update-badge {
        background-color: #4CAF50;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main application title - shown on all pages
if st.session_state.page == 'home':
    st.markdown("<h1 class='main-header'>PETRO-AFRO Advanced Analytics Framework</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Petroleum Engineering Analytics Suite</h2>", unsafe_allow_html=True)
else:
    st.title("PETRO-AFRO Advanced Analytics Framework")

# Display appropriate page based on session state
if st.session_state.page == 'home':
    # Information about the framework
    st.markdown("""
    This framework provides advanced analytics tools for petroleum engineering data analysis, 
    featuring machine learning models for well log analysis, seismic interpretation, and reservoir characterization.
    """)
    
    # Create three columns for the module cards
    col1, col2, col3 = st.columns(3)
    
    # Create custom large buttons
    with col1:
        st.markdown("<div class='big-button'>", unsafe_allow_html=True)
        if st.button("📊\nWell Log Analysis", key="well_log_button"):
            navigate_to('well_log')
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='button-description'>Analyze well log data, train models for porosity, permeability prediction, and lithology classification.</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='big-button'>", unsafe_allow_html=True)
        if st.button("🌊\nSeismic Analysis", key="seismic_button"):
            navigate_to('seismic')
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='button-description'>Process and interpret seismic data, identify geological structures, and train detection models.</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='big-button'>", unsafe_allow_html=True)
        if st.button("🧪\nReservoir Simulation", key="reservoir_button"):
            navigate_to('reservoir')
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='button-description'>Build reservoir models, run simulations, and optimize production strategies.</div>", unsafe_allow_html=True)
    
    # Getting Started Section
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("## Getting Started")
    
    st.markdown("<div class='steps-container'>", unsafe_allow_html=True)
    st.markdown("""
    1. Click on any module above to begin analysis
    2. Upload your dataset when prompted
    3. Follow the guided workflow for data analysis and model training
    4. Save your trained models for future use
    
    All modules support various formats of petroleum engineering data and provide 
    customizable machine learning models.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # New Updates Section
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Header with update count
    updates = read_updates()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## Recent Update Log")
    with col2:
        st.markdown(f"<div style='text-align: right;'><span class='update-badge'>{len(updates)} updates</span></div>", unsafe_allow_html=True)
    
    # Display updates in a container
    st.markdown("<div class='updates-container'>", unsafe_allow_html=True)
    
    # Show updates
    for update in updates:
        # Parse the update - format should be [YYYY-MM-DD] Update message
        if update.startswith("[") and "]" in update:
            date_str = update[1:update.find("]")]
            message = update[update.find("]")+1:].strip()
            
            # Check if this is a recent update (within the last 7 days)
            try:
                update_date = datetime.strptime(date_str, "%Y-%m-%d")
                days_ago = (datetime.now() - update_date).days
                is_new = days_ago <= 7
            except:
                is_new = False
            
            # Display the update with date highlighted
            if is_new:
                st.markdown(f"<div class='update-item'><span class='update-date'>[{date_str}]</span> {message} <span class='update-badge'>NEW</span></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='update-item'><span class='update-date'>[{date_str}]</span> {message}</div>", unsafe_allow_html=True)
        else:
            # Fallback for updates without date format
            st.markdown(f"<div class='update-item'>{update}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Import and display Well Log Analysis page
elif st.session_state.page == 'well_log':
    # Add home button
    if st.sidebar.button("🏠 Home"):
        navigate_to('home')
    
    st.sidebar.markdown("## Well Log Analysis")
    
    # Import the Well Log module
    try:
        # Try to dynamically import the module
        sys.path.append('.')
        import well_log_module
        well_log_module.show()
    except ImportError as e:
        st.error(f"Error importing well_log_module: {str(e)}")
        st.error("Please make sure well_log_module.py exists in the current directory")

# Seismic Analysis page
elif st.session_state.page == 'seismic':
    # Add home button
    if st.sidebar.button("🏠 Home"):
        navigate_to('home')
    
    st.header("Seismic Analysis")
    st.info("Seismic Analysis module is under development.")
    
    # Placeholder content for seismic module
    st.markdown("""
    ## Coming Soon: Seismic Data Analysis
    
    This module will include:
    - Seismic data visualization
    - Attribute analysis
    - Horizon and fault detection
    - Reservoir characterization
    """)

# Reservoir Simulation page
elif st.session_state.page == 'reservoir':
    # Add home button
    if st.sidebar.button("🏠 Home"):
        navigate_to('home')
    
    st.header("Reservoir Simulation")
    st.info("Reservoir Simulation module is under development.")
    
    # Placeholder content for reservoir module
    st.markdown("""
    ## Coming Soon: Reservoir Simulation
    
    This module will include:
    - Pressure-Volume-Temperature (PVT) analysis
    - Material balance calculations
    - Decline curve analysis
    - Production optimization
    """)

# Footer - shown on all pages
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("PETRO-AFRO Advanced Analytics Framework - Developed for petroleum engineering applications")