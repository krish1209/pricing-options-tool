import streamlit as st
from ui.pricing_view import render_pricing_tab
from ui.risk_view import render_risk_tab
from ui.market_view import render_market_tab
from ui.strategy_view import render_strategy_view
from ui.models_view import render_models_view
from utils.config import setup_sidebar
from utils.styling import load_theme
import os

def main():
    # Set page config
    st.set_page_config(
        page_title="Options Pricing & Risk Analysis",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply dark theme CSS
    load_theme()
    
    # Add custom header with logo
    st.markdown(
        """
        <div class="header">
            <div class="header-content">
                <h1>Options Pricing & Risk Analysis Tool</h1>
                <p>Analyze options using various pricing models and evaluate financial risk metrics</p>
            </div>
            <div class="header-links">
                <span style="color: var(--color-text-primary); margin-right: 15px;">Created by Krish Bagga</span>
                <a href="https://github.com/krish1209/pricing-options-tool" target="_blank" style="margin-right: 15px;">
                    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="38" />
                </a>
                <a href="https://www.linkedin.com/in/krishbagga/" target="_blank">
                    <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="35" />
                </a>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Set up the sidebar with common controls
    ticker, current_price, risk_free_rate, implied_vol = setup_sidebar()
    
    # Create tabs for the main sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Option Pricing", 
        "Risk Analysis", 
        "Market Data", 
        "Strategy Builder", 
        "Advanced Models"
    ])
    
    # Render each tab content
    with tab1:
        render_pricing_tab(ticker, current_price, risk_free_rate, implied_vol)
    
    with tab2:
        render_risk_tab(ticker, current_price, risk_free_rate, implied_vol)
    
    with tab3:
        render_market_tab(ticker, current_price)
    
    with tab4:
        render_strategy_view(ticker, current_price, risk_free_rate, implied_vol)
    
    with tab5:
        render_models_view(ticker, current_price, risk_free_rate, implied_vol)
    
    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>© 2025 Options Pricing & Risk Analysis Tool</p>
            <p>Built with Python, Streamlit, and Yahoo Finance</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Ensure the CSS directory exists
def setup_css_directory():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define CSS directory
    css_dir = os.path.join(current_dir, "static", "css", "dark")
    os.makedirs(css_dir, exist_ok=True)

if __name__ == "__main__":
    setup_css_directory()
    main()