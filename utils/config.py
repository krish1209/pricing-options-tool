import streamlit as st
from data.yahoo_finance import get_stock_data, get_risk_free_rate, calculate_implied_volatility

def setup_app():
    """Apply styling and setup the application header"""
    # Add your existing CSS styling here (from app.py)
    st.markdown("""
    <style>
        /* Color Palette */
        :root {
            --off-white: #F7F7F7;
            --amber: #FFA726;
            --brown: #704214;
            --black: #000000;
            --dark-gray: #121212;
            --medium-gray: #272727;
            --text-color: var(--off-white);
            --accent-color: var(--amber);
        }
        
        /* Global styles */
        .stApp {
            background-color: var(--black);
            color: var(--text-color) !important;
        }
        
        /* Force all text to be light by default */
        .stMarkdown, .stText, p, div, span, label {
            color: var(--text-color) !important;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: var(--accent-color) !important;
            font-weight: 600 !important;
        }
        
        h4, h5, h6 {
            color: var(--text-color) !important;
            font-weight: 500 !important;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: var(--dark-gray);
            border-right: 1px solid var(--medium-gray);
        }
        
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: var(--accent-color) !important;
        }
        
        section[data-testid="stSidebar"] button {
            background-color: var(--accent-color) !important;
            color: var(--black) !important;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: var(--accent-color) !important;
            color: var(--black) !important;
            border: none !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background-color: var(--brown) !important;
            color: var(--off-white) !important;
            box-shadow: 0 0 10px rgba(255, 167, 38, 0.5) !important;
        }
        
        /* Input text */
        .stTextInput > div > div > input {
            color: var(--off-white) !important;
            background-color: var(--medium-gray) !important;
            border: 1px solid var(--dark-gray) !important;
        }
        
        /* Fix number inputs */
        input[type="number"] {
            color: var(--off-white) !important;
            background-color: var(--medium-gray) !important;
            border: 1px solid var(--dark-gray) !important;
        }
        
        /* Radio buttons and checkboxes */
        .stRadio label, .stCheckbox label {
            color: var(--text-color) !important;
        }
        
        /* Radio buttons color */
        .stRadio [data-baseweb="radio"] div[role="radiogroup"] div[role="radio"] div {
            border-color: var(--accent-color) !important;
        }
        
        .stRadio [data-baseweb="radio"] div[role="radiogroup"] div[role="radio"][aria-checked="true"] div::after {
            background-color: var(--accent-color) !important;
        }
        
        /* Select boxes */
        .stSelectbox > div > div > div {
            color: var(--text-color) !important;
            background-color: var(--medium-gray) !important;
            border: 1px solid var(--dark-gray) !important;
        }
        
        /* Card-like elements */
        div.stDataFrame, div[data-testid="stTable"] {
            background-color: var(--dark-gray);
            border: 1px solid var(--medium-gray);
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1px;
            background-color: var(--dark-gray);
            border-radius: 4px 4px 0 0;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: var(--medium-gray);
            border-radius: 4px 4px 0 0;
            color: var(--off-white) !important;
            padding: 0.5rem 1rem;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--accent-color) !important;
            color: var(--black) !important;
            font-weight: 600 !important;
        }
        
        /* Tab content background */
        .stTabs [data-baseweb="tab-panel"] {
            background-color: var(--dark-gray);
            border-radius: 0 0 4px 4px;
            padding: 1rem;
            border: 1px solid var(--medium-gray);
            border-top: none;
        }
        
        /* Override the tab content color */
        .stTabs [data-baseweb="tab-panel"] > div {
            color: var(--text-color) !important;
        }
        
        /* Improved slider spacing */
        .stSlider {
            padding-top: 20px;
            padding-bottom: 35px;
            margin-bottom: 10px;
        }
        
        /* Improve slider label visibility */
        .stSlider label {
            margin-bottom: 8px !important;
            font-weight: 500 !important;
            font-size: 16px !important;
            color: var(--text-color) !important;
        }
        
        /* Better spacing for min/max values */
        .stSlider [data-baseweb] div[data-testid] {
            margin-top: 8px;
            color: var(--text-color) !important;
        }
        
        /* Fix slider track */
        .stSlider > div > div > div {
            background-color: var(--medium-gray) !important;
        }
        
        .stSlider > div > div > div > div {
            background-color: var(--accent-color) !important;
        }
        
        /* Fix slider thumb value positioning */
        .stSlider [data-testid="stThumbValue"] {
            position: absolute;
            background-color: var(--accent-color) !important;
            color: var(--black) !important;
            font-weight: 600 !important;
            padding: 3px 8px !important;
            border-radius: 4px !important;
            font-size: 13px !important;
            top: -28px !important;
            transform: translateX(-50%);
            z-index: 100;
        }
        
        /* Metrics styling */
        [data-testid="stMetricValue"] {
            color: var(--accent-color) !important;
            font-weight: bold !important;
        }
        
        [data-testid="stMetricDelta"] {
            background-color: var(--brown);
            padding: 2px 6px;
            border-radius: 4px;
            color: var(--off-white) !important;
        }
        
        /* Metric label color fix */
        [data-testid="stMetricLabel"] {
            color: var(--text-color) !important;
        }
        
        /* SelectBox styling */
        div[data-baseweb="select"] span {
            color: var(--text-color) !important;
        }
        
        /* MultiSelect options */
        div[role="listbox"] div {
            color: var(--text-color) !important;
            background-color: var(--medium-gray);
        }
        
        div[role="listbox"] div:hover {
            background-color: var(--dark-gray) !important;
        }
        
        /* Selected items in multiselect */
        div[data-baseweb="tag"] {
            background-color: var(--accent-color) !important;
        }
        
        div[data-baseweb="tag"] span {
            color: var(--black) !important;
        }
        
        /* Table layout improvements for better visibility */
        .dataframe {
            font-size: 13px !important;
            border-collapse: separate !important;
            border-spacing: 0 !important;
            width: 100% !important;
            border: none !important;
        }
        
        /* Ensure scrolling works properly */
        [data-testid="stDataFrame"] > div {
            max-height: 500px !important; /* Increased height */
            overflow: auto !important;
        }
        
        /* Ensure table cell text is light */
        .dataframe td {
            color: var(--text-color) !important;
            background-color: var(--dark-gray) !important;
        }
        
        /* Sticky header for tables */
        .dataframe thead th {
            position: sticky !important;
            top: 0 !important;
            z-index: 1 !important;
            background-color: var(--brown) !important;
            color: var(--off-white) !important;
            padding: 8px 4px !important;
            font-weight: 600 !important;
        }
        
        .dataframe td {
            background-color: var(--dark-gray);
            padding: 6px 4px !important;
            border-bottom: 1px solid var(--medium-gray) !important;
        }
        
        /* Better styling for call options */
        .call-options-table th {
            background-color: var(--brown) !important;
            color: var(--off-white) !important;
            padding: 8px 4px !important;
            text-align: center !important;
            font-weight: 600 !important;
            border-bottom: 1px solid var(--medium-gray) !important;
        }
        
        .call-options-table td {
            background-color: var(--dark-gray) !important;
            color: var(--text-color) !important;
            padding: 6px 4px !important;
            text-align: right !important;
            border-bottom: 1px solid var(--medium-gray) !important;
        }
        
        /* Better styling for put options */
        .put-options-table th {
            background-color: var(--brown) !important;
            color: var(--off-white) !important;
            padding: 8px 4px !important;
            text-align: center !important;
            font-weight: 600 !important;
            border-bottom: 1px solid var(--medium-gray) !important;
        }
        
        .put-options-table td {
            background-color: var(--dark-gray) !important;
            color: var(--text-color) !important;
            padding: 6px 4px !important;
            text-align: right !important;
            border-bottom: 1px solid var(--medium-gray) !important;
        }
        
        /* Special highlights for in-the-money options */
        .call-options-table .itm {
            background-color: rgba(255, 167, 38, 0.2) !important;
        }
        
        .put-options-table .itm {
            background-color: rgba(255, 167, 38, 0.2) !important;
        }
        
        /* Ensure option chain columns are spaced better */
        .options-container {
            display: flex;
            gap: 20px;
        }
        
        .option-column {
            flex: 1;
        }
        
        /* Custom header bar */
        .header-container {
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            padding: 1rem 0;
            border-bottom: 2px solid var(--accent-color);
            margin-bottom: 2rem;
        }
        
        .header-title {
            color: var(--accent-color);
            margin: 0;
            font-weight: 700;
        }
        
        .header-links {
            display: flex; 
            gap: 15px; 
            align-items: center;
        }
        
        .header-links img {
            transition: transform 0.2s ease;
            filter: invert(1);
        }
        
        .header-links img:hover {
            transform: scale(1.1);
        }
        
        /* Footer styling */
        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--dark-gray);
            color: var(--text-color);
            font-size: 0.9rem;
            text-align: center;
        }
        
        /* Interpretation text styling */
        .interpretation-text {
            background-color: var(--dark-gray);
            border-left: 3px solid var(--accent-color);
            padding: 0.8rem;
            border-radius: 0 4px 4px 0;
            margin: 1rem 0;
        }
        
        /* Result box styling */
        .result-box {
            background-color: var(--dark-gray);
            border: 1px solid var(--medium-gray);
            border-radius: 6px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }
        
        /* Plots and charts styling */
        [data-testid="stDecoration"] {
            background-color: var(--dark-gray) !important;
            border: 1px solid var(--medium-gray) !important;
            border-radius: 6px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Caption and legend text fixes */
        .js-plotly-plot .plotly .legend text, 
        .js-plotly-plot .plotly .xtick text, 
        .js-plotly-plot .plotly .ytick text {
            fill: var(--text-color) !important;
        }
        
        /* Fix code blocks */
        .stCodeBlock, code {
            background-color: var(--medium-gray) !important;
            color: var(--off-white) !important;
        }
        
        /* Misc elements fixes */
        .stProgress > div > div > div {
            background-color: var(--accent-color) !important;
        }
        
        /* Make sure all progress bars have proper contrast */
        .stProgress {
            color: var(--off-white) !important;
        }
        
        /* Fix expanders */
        .stExpander details {
            background-color: var(--dark-gray) !important;
            border: 1px solid var(--medium-gray) !important;
        }
        
        .stExpander details summary {
            color: var(--text-color) !important;
        }
        
        /* Fix captions */
        .stCaption {
            color: var(--text-color) !important;
            opacity: 0.8;
        }
        
        /* Fix info boxes */
        div[data-baseweb="notification"] {
            background-color: var(--dark-gray) !important;
            border-left-color: var(--accent-color) !important;
        }
        
        div[data-baseweb="notification"] div {
            color: var(--text-color) !important;
        }
        
        /* Fix tooltip text */
        div[data-baseweb="tooltip"] {
            background-color: var(--brown) !important;
            color: var(--off-white) !important;
        }
        
        /* Make plotly background dark */
        .js-plotly-plot .plotly {
            background-color: var(--dark-gray) !important;
        }
        
        .js-plotly-plot .plotly .main-svg {
            background-color: var(--dark-gray) !important;
        }
        
        /* Fix the main trace colors to be bright in dark mode */
        .js-plotly-plot .plotly .scatter .lines {
            stroke: var(--accent-color) !important;
        }
        
        /* Fix line chart grid */
        .js-plotly-plot .plotly .gridlayer path {
            stroke: var(--medium-gray) !important;
        }

        /* Fix axis lines */
        .js-plotly-plot .plotly .xaxis path.domain,
        .js-plotly-plot .plotly .yaxis path.domain {
            stroke: var(--text-color) !important;
        }
        /* Model selection multiselect button styling */
        div[data-baseweb="tag"] {
            background-color: #FFA726 !important; /* Amber color */
            border: none !important;
            border-radius: 4px !important;
            margin: 3px !important;
        }

        div[data-baseweb="tag"]:hover {
            background-color: #704214 !important; /* Brown color */
            box-shadow: 0 0 5px rgba(255, 167, 38, 0.5) !important;
        }

        div[data-baseweb="tag"] span {
            color: #000000 !important; /* Black text */
            font-weight: 500 !important;
        }

        div[data-baseweb="tag"] button, div[data-baseweb="tag"] svg {
            color: #000000 !important;
            fill: #000000 !important;
        }
        /* Model result columns styling */
        .model-results-column {
            background-color: #121212 !important;
            border-radius: 8px !important;
            border: 1px solid #272727 !important;
            padding: 15px !important;
            margin: 0 5px !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
        }

        .model-results-column h3 {
            border-bottom: 2px solid #FFA726 !important;
            padding-bottom: 8px !important;
            margin-bottom: 16px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Add your existing header from app.py
    st.markdown(
        """
        <div style="padding: 1rem 0; border-bottom: 2px solid #FFA726; margin-bottom: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h1 style="color: #FFA726; margin: 0; font-weight: 700;">Options Pricing & Risk Analysis Tool</h1>
                <div style="display: flex; gap: 15px; align-items: center;">
                    <a href="https://github.com/krish1209/pricing-options-tool" target="_blank">
                        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30" style="filter: invert(1);" />
                    </a>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <span style="color: #F7F7F7;">Created by - Krish Bagga</span>
                        <a href="https://www.linkedin.com/in/krishbagga/" target="_blank">
                            <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="20" style="filter: invert(1);" />
                        </a>
                    </div>
                </div>
            </div>
            <p style="color: #F7F7F7; margin-top: 0.5rem; margin-bottom: 0;">Analyze options using various pricing models and evaluate financial risk metrics.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def setup_sidebar():
    """Setup the sidebar and return common parameters"""
    st.sidebar.header("Settings")

    # Ticker input
    ticker = st.sidebar.text_input("Enter stock ticker:", "AAPL").upper()

    # Create a placeholder in the sidebar for stock data status
    stock_status = st.sidebar.empty()
    stock_status.info("Fetching stock data...")

    # Fetch stock data
    stock_data = get_stock_data(ticker)
    if not stock_data.empty:
        current_price = stock_data['Close'].iloc[-1]
        stock_status.success(f"Current {ticker} price: ${current_price:.2f}")
    else:
        stock_status.error(f"Failed to fetch data for {ticker}")
        current_price = 100.0  # Default for demonstration

    # Get risk-free rate
    risk_free_rate = get_risk_free_rate()
    st.sidebar.info(f"Current risk-free rate: {risk_free_rate*100:.2f}%")

    # Create a placeholder for volatility status
    vol_status = st.sidebar.empty()
    vol_status.info("Calculating volatility...")

    # Calculate implied volatility
    implied_vol = calculate_implied_volatility(ticker)
    if implied_vol:
        vol_status.info(f"Historical volatility (annualized): {implied_vol*100:.2f}%")
    else:
        implied_vol = 0.3  # Default
        vol_status.warning(f"Using default volatility: {implied_vol*100:.2f}%")
        
    return ticker, current_price, risk_free_rate, implied_vol