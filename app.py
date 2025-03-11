import streamlit as st
from ui.pricing_view import render_pricing_tab
from ui.risk_view import render_risk_tab
from ui.market_view import render_market_tab
from ui.strategy_view import render_strategy_view
from ui.models_view import render_models_view
from utils.config import setup_app, setup_sidebar

def main():
    # Set page config
    st.set_page_config(
        page_title="Options Pricing & Risk Analysis",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply styling and header
    setup_app()
    
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
        <div style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #272727; text-align: center;">
            <p style="color: #F7F7F7; font-size: 0.9rem;">© 2025 Options Pricing & Risk Analysis Tool | Built with Python, Streamlit, and Yahoo Finance</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()