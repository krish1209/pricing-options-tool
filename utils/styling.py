# In utils/styling.py

import streamlit as st
import os

def load_css(css_file_path):
    """
    Load and inject a CSS file into the Streamlit app
    
    Parameters:
    -----------
    css_file_path : str
        Path to the CSS file
    """
    with open(css_file_path, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def load_theme():
    """
    Load the dark theme CSS files
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to project root (assuming utils is one level below root)
    root_dir = os.path.dirname(current_dir)
    
    # Define the CSS directory
    css_dir = os.path.join(root_dir, "static", "css", "dark")
    
    # Create directory if it doesn't exist
    os.makedirs(css_dir, exist_ok=True)
    
    # Load CSS files in specific order (base styles first, then components)
    css_files = [
        "base.css",           # Base styles and variables
        "components.css",     # UI component styling
        "tables.css",         # Table and data display
        "charts.css",         # Chart and visualization overrides
        "animations.css"      # Animations and transitions
    ]
    
    # Load each file if it exists
    for css_file in css_files:
        file_path = os.path.join(css_dir, css_file)
        if os.path.exists(file_path):
            load_css(file_path)