import streamlit as st
import os
import zipfile
import io

st.set_page_config(page_title="Download & Resources", page_icon="📥", layout="wide")

st.title("📥 Download & Resources")

st.markdown("""
## Get the Application

This page provides download links and resources for the Wavelet Analysis application.
""")

# Create download functionality
def create_zip_file():
    """Create a zip file of the application."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add main app file
        if os.path.exists('app.py'):
            zip_file.write('app.py')
        
        # Add pages directory
        if os.path.exists('pages'):
            for root, dirs, files in os.walk('pages'):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, os.path.relpath(file_path))
        
        # Add requirements.txt
        if os.path.exists('requirements.txt'):
            zip_file.write('requirements.txt')
        
        # Add README.md
        if os.path.exists('README.md'):
            zip_file.write('README.md')
    
    zip_buffer.seek(0)
    return zip_buffer

# Download section
st.header("📦 Download Application")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Full Application Package
    
    Download the complete application as a ZIP file containing:
    - Main application file (`app.py`)
    - All page files (`pages/` directory)
    - Requirements file (`requirements.txt`)
    - Documentation (`README.md`)
    """)
    
    try:
        zip_file = create_zip_file()
        st.download_button(
            label="📥 Download Application (ZIP)",
            data=zip_file,
            file_name="wavelet_analysis_app.zip",
            mime="application/zip"
        )
    except Exception as e:
        st.error(f"Error creating zip file: {e}")
        st.info("You can manually zip the application files.")

with col2:
    st.markdown("""
    ### Installation Instructions
    
    1. **Extract the ZIP file** to your desired location
    
    2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    
    3. **Run the application**:
    ```bash
    streamlit run app.py
    ```
    
    4. **Open in browser**:
    The app will automatically open at `http://localhost:8501`
    """)

st.markdown("---")

# Individual file downloads
st.header("📄 Individual Files")

file_options = {
    "Main Application": "app.py",
    "Requirements": "requirements.txt",
    "README": "README.md"
}

for name, filepath in file_options.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
            st.download_button(
                label=f"Download {name}",
                data=content,
                file_name=filepath,
                mime="text/plain" if filepath.endswith('.txt') or filepath.endswith('.md') else "text/x-python",
                key=f"download_{filepath}"
            )

st.markdown("---")

# Resources section
st.header("📚 Additional Resources")

st.markdown("""
### Recommended Reading

1. **Books**:
   - "Wavelets and Wavelet Transforms" by C. Sidney Burrus
   - "A Wavelet Tour of Signal Processing" by Stephane Mallat
   - "Introduction to Wavelets and Wavelet Transforms" by C. Sidney Burrus et al.

2. **Online Resources**:
   - [PyWavelets Documentation](https://pywavelets.readthedocs.io/)
   - [Wavelet Tutorial](http://users.rowan.edu/~polikar/WTtutorial.html)
   - [MathWorks Wavelet Toolbox](https://www.mathworks.com/products/wavelet.html)

3. **Python Libraries**:
   - **PyWavelets**: Main wavelet library (`pip install PyWavelets`)
   - **scipy.signal**: Additional signal processing tools
   - **matplotlib**: Visualization
   - **numpy**: Numerical computations

### Code Repositories

- **PyWavelets GitHub**: https://github.com/PyWavelets/pywt
- **Example Notebooks**: Check PyWavelets documentation for Jupyter notebook examples

### Video Tutorials

- Search YouTube for "wavelet transform tutorial"
- "Introduction to Wavelets" by various educational channels
- Signal processing courses on Coursera, edX

### Research Papers

- Daubechies, I. (1992). "Ten Lectures on Wavelets"
- Mallat, S. (1989). "A Theory for Multiresolution Signal Decomposition"
- Many papers available on arXiv.org

---

## System Requirements

### Minimum Requirements
- **Python**: 3.7 or higher
- **RAM**: 2 GB minimum (4 GB recommended)
- **Disk Space**: 100 MB for application + dependencies
- **OS**: Windows, macOS, or Linux

### Recommended Requirements
- **Python**: 3.9 or higher
- **RAM**: 8 GB or more
- **CPU**: Multi-core processor for faster computations
- **Browser**: Modern browser (Chrome, Firefox, Edge, Safari)

### Dependencies

The application requires the following Python packages:
- `streamlit` - Web framework
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `PyWavelets` - Wavelet transforms
- `scipy` - Signal processing

All dependencies are listed in `requirements.txt`.

---

## Troubleshooting

### Common Issues

**Issue**: Import errors
- **Solution**: Make sure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: App won't start
- **Solution**: Check Python version (3.7+), ensure Streamlit is installed correctly

**Issue**: Slow performance
- **Solution**: Reduce signal length, use simpler wavelets (haar, db2), reduce decomposition levels

**Issue**: Memory errors
- **Solution**: Process signals in chunks, reduce visualization complexity

### Getting Help

1. Check the [Code Explanation page](?page=7_Code_Explanation) for implementation details
2. Review PyWavelets documentation for API reference
3. Check Streamlit documentation for UI issues
4. Search Stack Overflow for specific error messages

---

## License and Usage

This application is provided for educational purposes. Feel free to:
- ✅ Use it for learning
- ✅ Modify and adapt for your needs
- ✅ Share with others
- ✅ Use in educational settings

**Note**: This is an educational tool. For production applications, ensure proper testing and validation.

---

## Version Information

- **Application Version**: 1.0
- **Created**: 2024
- **Last Updated**: 2024
- **Python Version**: 3.7+
- **Streamlit Version**: 1.0+

---

## Feedback and Contributions

If you find bugs or have suggestions for improvements:
1. Check existing documentation
2. Review code comments
3. Test with different inputs
4. Document any issues you find

Thank you for using the Wavelet Analysis Application!
""")


