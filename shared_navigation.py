"""
Shared navigation component for all pages
"""
import streamlit as st

def render_navigation():
    """Render the main navigation sidebar that appears on all pages"""
    
    # Hide Streamlit's automatic navigation
    st.markdown("""
    <style>
        /* Hide Streamlit's automatic page navigation completely */
        [data-testid="stSidebarNav"],
        nav[data-testid="stSidebarNav"],
        [data-testid="stSidebar"] nav,
        [data-testid="stSidebar"] ul[data-testid="stSidebarNav"],
        .css-1d391kg nav,
        section[data-testid="stSidebar"] > div:nth-child(2) nav,
        section[data-testid="stSidebar"] > div nav,
        [data-testid="stSidebar"] > div > nav,
        [data-testid="stSidebar"] [role="navigation"],
        nav[role="navigation"],
        section[data-testid="stSidebar"] section nav,
        section[data-testid="stSidebar"] > section > nav,
        div[data-testid="stSidebar"] nav,
        [data-testid="stSidebar"] > div:has(nav),
        [data-testid="stSidebar"] > section:has(nav) {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            overflow: hidden !important;
            opacity: 0 !important;
            position: absolute !important;
            left: -9999px !important;
        }
        
        [data-testid="stSidebar"] a[href*="pages"],
        [data-testid="stSidebar"] a[href*="page"],
        [data-testid="stSidebar"] a[href*="/1_"],
        [data-testid="stSidebar"] a[href*="/2_"],
        [data-testid="stSidebar"] a[href*="/3_"] {
            display: none !important;
        }
        
        [data-testid="stSidebar"] button[key^="nav_btn_"] {
            display: block !important;
            visibility: visible !important;
        }
    </style>
    
    <script>
        function hideAutoNavigation() {
            const selectors = [
                '[data-testid="stSidebarNav"]',
                'nav[data-testid="stSidebarNav"]',
                '[data-testid="stSidebar"] nav',
                '[data-testid="stSidebar"] ul[data-testid="stSidebarNav"]',
                'section[data-testid="stSidebar"] > div nav',
                '[data-testid="stSidebar"] [role="navigation"]'
            ];
            
            selectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    el.style.display = 'none';
                    el.style.visibility = 'hidden';
                    el.style.height = '0';
                    el.style.overflow = 'hidden';
                });
            });
        }
        
        hideAutoNavigation();
        window.addEventListener('load', hideAutoNavigation);
        window.addEventListener('DOMContentLoaded', hideAutoNavigation);
        
        const observer = new MutationObserver(hideAutoNavigation);
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        setInterval(hideAutoNavigation, 500);
    </script>
    """, unsafe_allow_html=True)
    
    # Navigation header
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

