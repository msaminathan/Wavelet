import streamlit as st
from shared_navigation import render_navigation

st.set_page_config(page_title="Code Explanation", page_icon="💻", layout="wide")

# Render shared navigation
render_navigation()

st.title("💻 Code Explanation")

st.markdown("""
## Understanding the Implementation

This page explains the key code components used throughout this application.
""")

# Code sections
code_sections = st.tabs([
    "Wavelet Decomposition",
    "Signal Reconstruction",
    "Denoising Algorithm",
    "Visualization Code",
    "Multi-Page Structure"
])

with code_sections[0]:
    st.subheader("1. Wavelet Decomposition")
    st.markdown("""
    The core operation in wavelet analysis is decomposing a signal into different resolution levels.
    """)
    
    st.code("""
import numpy as np
import pywt

def wavelet_decomposition(signal, wavelet='db4', level=4):
    \"\"\"
    Decompose a signal using Discrete Wavelet Transform (DWT).
    
    Parameters:
    -----------
    signal : array_like
        Input signal to decompose
    wavelet : str
        Wavelet name (e.g., 'db4', 'haar', 'coif2')
    level : int
        Number of decomposition levels
        
    Returns:
    --------
    coeffs : list
        List of coefficient arrays [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        where cA is approximation and cD are details
    \"\"\"
    # Perform decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # coeffs structure:
    # [cA_level, cD_level, cD_{level-1}, ..., cD_1]
    # cA = approximation (low frequency)
    # cD = details (high frequency)
    
    return coeffs

# Example usage
t = np.linspace(0, 1, 1000)
x = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*30*t)
coeffs = wavelet_decomposition(x, 'db4', level=4)

print(f"Number of levels: {len(coeffs)-1}")
print(f"Approximation length: {len(coeffs[0])}")
print(f"Detail lengths: {[len(c) for c in coeffs[1:]]}")
    """, language='python')
    
    st.markdown("""
    **Key Points:**
    - `pywt.wavedec()` performs the forward DWT
    - Returns a list: first element is approximation, rest are details
    - Each level has half the samples of the previous level
    - Total number of coefficients equals input length (for orthogonal wavelets)
    """)

with code_sections[1]:
    st.subheader("2. Signal Reconstruction")
    st.markdown("""
    Reconstructing the original signal from wavelet coefficients.
    """)
    
    st.code("""
def wavelet_reconstruction(coeffs, wavelet='db4'):
    \"\"\"
    Reconstruct signal from wavelet coefficients.
    
    Parameters:
    -----------
    coeffs : list
        Wavelet coefficients from decomposition
    wavelet : str
        Wavelet name (must match decomposition)
        
    Returns:
    --------
    reconstructed : array
        Reconstructed signal
    \"\"\"
    # Reconstruct from coefficients
    reconstructed = pywt.waverec(coeffs, wavelet)
    
    # Note: Length might differ slightly due to padding
    # In practice, you may need to trim:
    # if len(reconstructed) > original_length:
    #     reconstructed = reconstructed[:original_length]
    
    return reconstructed

# Perfect reconstruction property
coeffs = pywt.wavedec(x, 'db4', level=4)
x_recon = pywt.waverec(coeffs, 'db4')

# Check reconstruction error
error = np.abs(x - x_recon[:len(x)])
print(f"Reconstruction RMSE: {np.sqrt(np.mean(error**2))}")
# Should be very small (numerical precision)
    """, language='python')
    
    st.markdown("""
    **Key Points:**
    - `pywt.waverec()` performs inverse DWT
    - Perfect reconstruction: original signal can be exactly recovered
    - Must use same wavelet for decomposition and reconstruction
    - Length may differ slightly due to padding (pywt handles this)
    """)

with code_sections[2]:
    st.subheader("3. Denoising Algorithm")
    st.markdown("""
    Wavelet-based denoising using thresholding.
    """)
    
    st.code("""
def wavelet_denoise(signal, wavelet='db4', threshold_type='soft', 
                    threshold_mode='universal', level=4):
    \"\"\"
    Denoise signal using wavelet thresholding.
    
    Parameters:
    -----------
    signal : array_like
        Noisy input signal
    wavelet : str
        Wavelet to use
    threshold_type : str
        'soft' or 'hard' thresholding
    threshold_mode : str
        'universal' uses sqrt(2*log(N)) rule
        'sure' uses SURE (Stein's Unbiased Risk Estimate)
    level : int
        Decomposition level
        
    Returns:
    --------
    denoised : array
        Denoised signal
    \"\"\"
    # Decompose signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Estimate noise level from finest detail coefficients
    # Median absolute deviation (MAD) estimator
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # Calculate threshold
    if threshold_mode == 'universal':
        # Universal threshold: sqrt(2*log(N))
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    else:
        # SURE threshold (more sophisticated)
        threshold = pywt.threshold(coeffs[-1], mode='sure')
    
    # Apply thresholding to detail coefficients (not approximation)
    coeffs_thresh = [coeffs[0]]  # Keep approximation
    for c in coeffs[1:]:  # Threshold details
        coeffs_thresh.append(pywt.threshold(c, threshold, threshold_type))
    
    # Reconstruct
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    
    return denoised[:len(signal)]

# Why this works:
# 1. Signal has structure → large coefficients
# 2. Noise is random → many small coefficients
# 3. Thresholding removes small coefficients → removes noise
# 4. Large coefficients preserved → signal preserved
    """, language='python')
    
    st.markdown("""
    **Key Points:**
    - Noise estimation from finest detail coefficients
    - Universal threshold: sqrt(2*log(N)) is optimal for Gaussian noise
    - Soft thresholding: shrinks coefficients (better for smooth signals)
    - Hard thresholding: sets small coefficients to zero (better for sparse signals)
    - Approximation coefficients usually not thresholded
    """)

with code_sections[3]:
    st.subheader("4. Visualization Code")
    st.markdown("""
    Creating informative visualizations for wavelet analysis.
    """)
    
    st.code("""
import matplotlib.pyplot as plt
import numpy as np

def plot_wavelet_decomposition(signal, coeffs, t, wavelet_name):
    \"\"\"
    Visualize wavelet decomposition.
    
    Parameters:
    -----------
    signal : array
        Original signal
    coeffs : list
        Wavelet coefficients
    t : array
        Time axis
    wavelet_name : str
        Name of wavelet used
    \"\"\"
    fig, axes = plt.subplots(len(coeffs) + 1, 1, figsize=(12, 2*(len(coeffs)+1)))
    
    # Plot original signal
    axes[0].plot(t, signal, 'b-', linewidth=2)
    axes[0].set_title('Original Signal', fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot each level
    level_names = ['Approximation'] + [f'Detail Level {i}' 
                                       for i in range(len(coeffs)-1, 0, -1)]
    
    y_offset = 0
    for i, (coeff, name) in enumerate(zip(coeffs, level_names)):
        t_coeff = np.linspace(t[0], t[-1], len(coeff))
        axes[i+1].plot(t_coeff, coeff + y_offset, linewidth=1.5, label=name)
        axes[i+1].set_ylabel('Amplitude')
        axes[i+1].legend()
        axes[i+1].grid(True, alpha=0.3)
        y_offset += np.max(np.abs(coeff)) * 2
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    return fig

def plot_scalogram(coeffs_cwt, scales, t, title='Scalogram'):
    \"\"\"
    Plot continuous wavelet transform scalogram.
    
    Parameters:
    -----------
    coeffs_cwt : 2D array
        CWT coefficients [scales x time]
    scales : array
        Scale values
    t : array
        Time axis
    title : str
        Plot title
    \"\"\"
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot as image
    coeffs_cwt_abs = np.abs(coeffs_cwt)
    # Ensure 2D array for imshow
    if coeffs_cwt_abs.ndim == 1:
        coeffs_cwt_abs = coeffs_cwt_abs.reshape(1, -1)
    elif coeffs_cwt_abs.ndim == 0:
        coeffs_cwt_abs = np.array([[coeffs_cwt_abs]])
    im = ax.imshow(coeffs_cwt_abs, aspect='auto', cmap='jet',
                   extent=[t[0], t[-1], scales[-1], scales[0]])
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Scale')
    ax.set_title(title, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Magnitude')
    
    return fig

# Example: Convert coefficients to array for visualization
def coeffs_to_image(coeffs):
    \"\"\"
    Convert wavelet coefficients to 2D array for visualization.
    \"\"\"
    coeffs_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)
    return coeffs_arr
    """, language='python')
    
    st.markdown("""
    **Key Points:**
    - Use `pywt.coeffs_to_array()` to convert coefficient list to 2D array
    - Scalograms show time-frequency content
    - Offset coefficients vertically for clarity
    - Use appropriate colormaps (jet, viridis, hot) for different data types
    """)

with code_sections[4]:
    st.subheader("5. Multi-Page Streamlit Structure")
    st.markdown("""
    How this multi-page application is organized.
    """)
    
    st.code("""
# Directory Structure
Wavelet/
├── app.py                    # Main entry point
├── pages/
│   ├── 1_Introduction.py
│   ├── 2_Why_Wavelets.py
│   ├── 3_Mathematical_Foundation.py
│   ├── 4_Wavelet_Examples.py
│   ├── 5_Interactive_Visualization.py
│   ├── 6_Engineering_Applications.py
│   ├── 7_Code_Explanation.py
│   ├── 8_Download_Resources.py
│   ├── 9_About_Cursor_AI.py
│   └── 10_Additional_Thoughts.py
├── requirements.txt
└── README.md

# app.py (Main file)
import streamlit as st

st.set_page_config(
    page_title="Wavelet Analysis",
    page_icon="🌊",
    layout="wide"
)

st.title("Wavelet Analysis & Applications")
# Main page content

# Each page file (e.g., pages/1_Introduction.py)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Introduction", page_icon="📚")

st.title("Introduction to Wavelets")
# Page-specific content

# Streamlit automatically:
# 1. Detects files in pages/ directory
# 2. Creates navigation sidebar
# 3. Numbers pages based on filename prefix
    """, language='python')
    
    st.markdown("""
    **Key Points:**
    - Streamlit automatically creates multi-page apps from `pages/` directory
    - File names with numbers control page order
    - Each page is independent (can import different libraries)
    - `st.set_page_config()` sets page-specific metadata
    """)

st.markdown("""
---

## Common Patterns and Best Practices

### 1. Error Handling

```python
try:
    coeffs = pywt.wavedec(signal, wavelet, level=level)
except ValueError as e:
    st.error(f"Invalid wavelet or level: {e}")
    return None
```

### 2. Length Matching

```python
# Wavelet transform may change signal length
x_recon = pywt.waverec(coeffs, wavelet)
if len(x_recon) != len(x_original):
    x_recon = x_recon[:len(x_original)]  # Trim
    # or
    x_original = np.pad(x_original, (0, len(x_recon) - len(x_original)))  # Pad
```

### 3. Memory Efficiency

```python
# For large signals, process in chunks
chunk_size = 10000
for i in range(0, len(signal), chunk_size):
    chunk = signal[i:i+chunk_size]
    coeffs_chunk = pywt.wavedec(chunk, wavelet, level=level)
    # Process chunk
```

### 4. Choosing Wavelet and Level

```python
# Level selection: don't exceed max level
max_level = pywt.dwt_max_level(len(signal), wavelet.dec_len)
level = min(requested_level, max_level)

# Wavelet selection based on application:
# - Denoising: db4, db8 (smooth)
# - Compression: bior2.2, bior4.4 (symmetric)
# - Edge detection: haar, db2 (sharp)
# - General: db4, coif2 (balanced)
```

### 5. Performance Optimization

```python
# Use appropriate wavelets for speed
# Fast: haar, db2
# Medium: db4, db8
# Slow: coiflets, biorthogonal

# For real-time applications, pre-compute filter coefficients
wavelet = pywt.Wavelet('db4')
# Use wavelet.dec_lo, wavelet.dec_hi for custom implementation
```

## Summary

The code in this application follows these principles:
1. **Modularity**: Each function has a single purpose
2. **Error Handling**: Graceful failure with informative messages
3. **Documentation**: Clear docstrings and comments
4. **Efficiency**: Appropriate wavelet and level selection
5. **Visualization**: Clear, informative plots

Understanding these patterns will help you implement wavelet analysis in your own projects!
""")


