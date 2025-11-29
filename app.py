import streamlit as st

st.set_page_config(
    page_title="Wavelet Analysis & Applications",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for vibrant styling
st.markdown("""
<style>
    /* Main background - clean white */
    .stApp {
        background: #ffffff;
    }
    
    /* Main content area */
    .main .block-container {
        background: #ffffff;
        padding: 2rem;
        margin-top: 2rem;
    }
    
    /* Vibrant title styling */
    h1 {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling - clean background */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    [data-testid="stSidebar"] {
        background: #f8f9fa;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #333 !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: #555 !important;
    }
    
    /* Cards and containers */
    .stMarkdown {
        background: transparent;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(90deg, #764ba2, #667eea);
    }
    
    /* Headers */
    h2 {
        color: #667eea;
        border-left: 5px solid #f093fb;
        padding-left: 1rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #764ba2;
        margin-top: 1.5rem;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #f093fb, #4facfe);
        color: white;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border-left: 4px solid #667eea;
        border-radius: 10px;
    }
    
    /* Success boxes */
    .stSuccess {
        background: linear-gradient(135deg, #4facfe15, #00f2fe15);
        border-left: 4px solid #4facfe;
        border-radius: 10px;
    }
    
    /* Warning boxes */
    .stWarning {
        background: linear-gradient(135deg, #f093fb15, #f5576c15);
        border-left: 4px solid #f093fb;
        border-radius: 10px;
    }
    
    /* Error boxes */
    .stError {
        background: linear-gradient(135deg, #f5576c15, #f093fb15);
        border-left: 4px solid #f5576c;
        border-radius: 10px;
    }
    
    /* Hide Streamlit's automatic page navigation completely */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Hide the default page navigation if it appears */
    .css-1d391kg nav {
        display: none !important;
    }
    
    /* Hide any ul/li navigation lists */
    [data-testid="stSidebar"] ul[data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Hide navigation in sidebar */
    nav[data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Additional selector to catch all navigation */
    section[data-testid="stSidebar"] > div:nth-child(2) nav {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Vibrant header with gradient
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe, #00f2fe);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;
               font-size: 3.5rem;
               font-weight: 900;
               margin: 0;
               text-shadow: 3px 3px 6px rgba(0,0,0,0.2);">
        🌊 Wavelet Analysis & Applications
    </h1>
    <p style="font-size: 1.3rem; color: #667eea; font-weight: 600; margin-top: 0.5rem;">
        A Comprehensive Guide to Understanding Wavelets
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea15, #764ba215);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            border: 2px solid #667eea30;">
    <h2 style="color: #667eea; text-align: center; margin: 0 0 1rem 0; font-size: 1.8rem;">
        🧭 Navigation
    </h2>
</div>
""", unsafe_allow_html=True)

# Navigation items - bold text links without background
nav_items = [
    ("📚", "Introduction", "pages/1_Introduction.py"),
    ("❓", "Why Wavelets?", "pages/2_Why_Wavelets.py"),
    ("🔢", "Mathematical Foundation", "pages/3_Mathematical_Foundation.py"),
    ("🎨", "Wavelet Examples", "pages/4_Wavelet_Examples.py"),
    ("🎮", "Interactive Visualization", "pages/5_Interactive_Visualization.py"),
    ("🔧", "Engineering Applications", "pages/6_Engineering_Applications.py"),
    ("💻", "Code Explanation", "pages/7_Code_Explanation.py"),
    ("📥", "Download & Resources", "pages/8_Download_Resources.py"),
    ("🤖", "About Cursor AI", "pages/9_About_Cursor_AI.py"),
    ("💭", "Additional Thoughts", "pages/10_Additional_Thoughts.py")
]

for icon, title, page_path in nav_items:
    # Create a button that navigates to the page
    if st.sidebar.button(f"{icon} {title}", key=f"nav_btn_{page_path}", use_container_width=True):
        st.switch_page(page_path)
    
    # Add custom styling for bold text without background
    st.sidebar.markdown(f"""
    <style>
        button[data-testid="baseButton-secondary"][key="nav_btn_{page_path}"] {{
            background: transparent !important;
            border: none !important;
            border-radius: 5px !important;
            text-align: left !important;
            padding: 0.6rem 0.8rem !important;
            color: #333 !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            box-shadow: none !important;
            margin-bottom: 0.3rem !important;
        }}
        button[data-testid="baseButton-secondary"][key="nav_btn_{page_path}"]:hover {{
            background: #f0f0f0 !important;
            color: #667eea !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# Welcome section with vibrant cards
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea15, #764ba215);
            padding: 2rem;
            border-radius: 20px;
            margin: 2rem 0;
            border: 2px solid #667eea40;">
    <h2 style="color: #667eea; text-align: center; font-size: 2.5rem; margin-bottom: 1rem;">
        ✨ Welcome to the Wavelet Analysis App! ✨
    </h2>
    <p style="font-size: 1.2rem; text-align: center; color: #555; line-height: 1.8;">
        This multi-page application provides a comprehensive exploration of wavelets, their mathematical foundations, 
        visualizations, and practical applications in engineering and signal processing.
    </p>
</div>
""", unsafe_allow_html=True)

# What are Wavelets card
st.markdown("""
<div style="background: linear-gradient(135deg, #f093fb15, #4facfe15);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            border-left: 5px solid #f093fb;">
    <h3 style="color: #764ba2; margin-top: 0;">🌊 What are Wavelets?</h3>
    <p style="font-size: 1.1rem; color: #555; line-height: 1.8;">
        Wavelets are mathematical functions that decompose signals into different frequency components, 
        similar to Fourier transforms, but with the added advantage of <strong style="color: #667eea;">time localization</strong>. 
        Unlike Fourier transforms that provide frequency information for the entire signal, wavelets can analyze both 
        frequency and time simultaneously.
    </p>
</div>
""", unsafe_allow_html=True)

# Key Features with colorful cards
st.markdown("### 🎯 Key Features of This App")
col1, col2 = st.columns(2)

features = [
    ("📚", "Educational Content", "From basics to advanced concepts", "#667eea"),
    ("🎨", "Interactive Visualizations", "Explore wavelets dynamically", "#764ba2"),
    ("💻", "Code Examples", "Learn by seeing and running code", "#f093fb"),
    ("🔧", "Engineering Applications", "Real-world use cases", "#4facfe"),
    ("📥", "Downloadable Resources", "Take the app with you", "#00f2fe"),
    ("🤖", "AI-Powered Creation", "Built with Cursor AI", "#f5576c")
]

for i, (icon, title, desc, color) in enumerate(features):
    with (col1 if i % 2 == 0 else col2):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}20, {color}10);
                    padding: 1.2rem;
                    border-radius: 12px;
                    margin: 0.8rem 0;
                    border: 2px solid {color}40;
                    transition: transform 0.3s ease;
                    box-shadow: 0 4px 10px {color}20;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 2rem;">{icon}</span>
                <div>
                    <h4 style="color: {color}; margin: 0; font-size: 1.1rem;">{title}</h4>
                    <p style="color: #666; margin: 0.3rem 0 0 0; font-size: 0.95rem;">{desc}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Getting Started section
st.markdown("""
<div style="background: linear-gradient(135deg, #fee14020, #fa709a20);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 2rem 0;
            border: 2px solid #fee14060;">
    <h3 style="color: #fa709a; margin-top: 0;">🚀 Getting Started</h3>
    <p style="font-size: 1.1rem; color: #555; line-height: 1.8;">
        Use the sidebar to navigate through different sections. Start with the <strong style="color: #667eea;">Introduction</strong> page 
        to begin your journey into the world of wavelets!
    </p>
</div>
""", unsafe_allow_html=True)

# Installation instructions with code block styling
st.markdown("""
<div style="background: linear-gradient(135deg, #30cfd020, #33086720);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            border-left: 5px solid #30cfd0;">
    <h4 style="color: #330867; margin-top: 0;">📦 Installation Instructions</h4>
    <p style="color: #555; margin-bottom: 1rem;">
        This app requires Python packages listed in <code style="background: #667eea20; padding: 0.2rem 0.5rem; border-radius: 5px;">requirements.txt</code>
    </p>
    <div style="background: #1e1e1e; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <code style="color: #4ec9b0;">
            pip install -r requirements.txt<br>
            streamlit run app.py
        </code>
    </div>
</div>
""", unsafe_allow_html=True)

