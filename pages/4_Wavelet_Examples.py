import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pywt
from shared_navigation import render_navigation

st.set_page_config(page_title="Wavelet Examples", page_icon="🎨", layout="wide")

# Render shared navigation
render_navigation()

# Set vibrant matplotlib style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#f5576c'])

# Vibrant page styling
st.markdown("""
<style>
    h1 {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 style="background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;
               font-size: 2.8rem;
               font-weight: 800;
               margin: 0;">
        🎨 Wavelet Examples: How Wavelets Look
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #f093fb20, #4facfe20);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 2px solid #f093fb40;">
    <h2 style="color: #764ba2; margin-top: 0; text-align: center;">🌈 Visualizing Different Wavelet Families</h2>
    <p style="font-size: 1.1rem; color: #555; text-align: center; margin: 0.5rem 0 0 0;">
        This page shows various wavelets and their properties. Each wavelet has unique characteristics 
        that make it suitable for different applications.
    </p>
</div>
""", unsafe_allow_html=True)

# Wavelet selection
wavelet_families = {
    'Haar': 'haar',
    'Daubechies (db4)': 'db4',
    'Daubechies (db8)': 'db8',
    'Coiflet (coif2)': 'coif2',
    'Biorthogonal (bior2.2)': 'bior2.2',
    'Symlet (sym4)': 'sym4',
    'Morlet': 'morl',
    'Mexican Hat': 'mexh'
}

selected = st.selectbox("Select a Wavelet to Visualize", list(wavelet_families.keys()))

wavelet_name = wavelet_families[selected]

# Generate visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

try:
    if wavelet_name in ['morl', 'mexh']:
        # Continuous wavelets
        t = np.linspace(-4, 4, 1000)
        if wavelet_name == 'morl':
            # Morlet wavelet approximation
            psi = np.exp(-t**2/2) * np.cos(5*t)
        else:  # mexh
            # Mexican hat wavelet
            psi = (1 - t**2) * np.exp(-t**2/2)
        
        axes[0, 0].plot(t, psi, color='#667eea', linewidth=2.5)
        axes[0, 0].set_title(f'{selected} Wavelet', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Frequency domain
        psi_fft = np.fft.fft(psi)
        freq = np.fft.fftfreq(len(t), t[1] - t[0])
        axes[0, 1].plot(freq[:len(freq)//2], np.abs(psi_fft[:len(freq)//2]), color='#f093fb', linewidth=2.5)
        axes[0, 1].set_title('Frequency Domain', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        
    else:
        # Discrete wavelets
        try:
            wavelet = pywt.Wavelet(wavelet_name)
            phi, psi, x = wavelet.wavefun(level=5)
            
            axes[0, 0].plot(x, psi, color='#667eea', linewidth=2.5, label='Wavelet')
            axes[0, 0].plot(x, phi, color='#f093fb', linewidth=2.5, linestyle='--', label='Scaling Function')
            axes[0, 0].set_title(f'{selected} Wavelet & Scaling Function', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            
            # Frequency domain
            psi_fft = np.fft.fft(psi)
            freq = np.fft.fftfreq(len(psi), x[1] - x[0])
            axes[0, 1].plot(freq[:len(freq)//2], np.abs(psi_fft[:len(freq)//2]), color='#4facfe', linewidth=2.5)
            axes[0, 1].set_title('Frequency Domain', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Frequency')
            axes[0, 1].set_ylabel('Magnitude')
            axes[0, 1].grid(True, alpha=0.3)
            
        except Exception as e:
            st.error(f"Error generating wavelet: {e}")
            axes[0, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # Show scaled versions
    t_scale = np.linspace(-4, 4, 1000)
    scales = [1, 2, 4]
    colors = ['b', 'g', 'r']
    
    for i, scale in enumerate(scales):
        if wavelet_name == 'morl':
            psi_scaled = np.exp(-(t_scale/scale)**2/2) * np.cos(5*t_scale/scale) / np.sqrt(scale)
        elif wavelet_name == 'mexh':
            psi_scaled = (1 - (t_scale/scale)**2) * np.exp(-(t_scale/scale)**2/2) / np.sqrt(scale)
        else:
            try:
                wavelet = pywt.Wavelet(wavelet_name)
                _, psi_base, x_base = wavelet.wavefun(level=5)
                # Interpolate for scaling
                from scipy.interpolate import interp1d
                f = interp1d(x_base, psi_base, kind='linear', fill_value=0, bounds_error=False)
                psi_scaled = f(t_scale / scale) / np.sqrt(scale)
            except:
                psi_scaled = np.zeros_like(t_scale)
        
        vibrant_colors = ['#667eea', '#764ba2', '#f093fb']
        axes[1, 0].plot(t_scale, psi_scaled, color=vibrant_colors[i], linewidth=2, label=f'Scale a={scale}')
    
    axes[1, 0].set_title('Scaled Wavelets', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Example: Decomposing a signal
    t_signal = np.linspace(0, 1, 1000)
    signal = np.sin(2*np.pi*10*t_signal) * (t_signal < 0.5) + np.sin(2*np.pi*50*t_signal) * (t_signal >= 0.5)
    
    if wavelet_name not in ['morl', 'mexh']:
        try:
            coeffs = pywt.wavedec(signal, wavelet_name, level=4)
            axes[1, 1].plot(t_signal, signal, 'k-', linewidth=1, alpha=0.5, label='Original')
            y_offset = 0
            for i, coeff in enumerate(coeffs):
                t_coeff = np.linspace(0, 1, len(coeff))
                axes[1, 1].plot(t_coeff, coeff + y_offset, linewidth=1.5, label=f'Level {len(coeffs)-1-i}')
                y_offset += np.max(np.abs(coeff)) * 2
            axes[1, 1].set_title('Wavelet Decomposition', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Amplitude (offset)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        except:
            axes[1, 1].plot(t_signal, signal, 'b-', linewidth=2)
            axes[1, 1].set_title('Example Signal', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Amplitude')
            axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].plot(t_signal, signal, 'b-', linewidth=2)
        axes[1, 1].set_title('Example Signal', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"Error: {e}")

st.markdown(f"""
### Code to Generate {selected} Wavelet

```python
import numpy as np
import matplotlib.pyplot as plt
import pywt

# For discrete wavelets
wavelet = pywt.Wavelet('{wavelet_name}')
phi, psi, x = wavelet.wavefun(level=5)

plt.figure(figsize=(12, 4))
plt.plot(x, psi, 'b-', linewidth=2, label='Wavelet')
plt.plot(x, phi, 'r--', linewidth=2, label='Scaling Function')
plt.title('{selected} Wavelet')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Properties
print(f"Wavelet: {{wavelet.name}}")
print(f"Family: {{wavelet.family_name}}")
print(f"Vanishing moments: {{wavelet.vanishing_moments_psi}}")
print(f"Support width: {{wavelet.dec_len}}")
```

### Wavelet Properties

""")

# Display properties if available
if wavelet_name not in ['morl', 'mexh']:
    try:
        wavelet = pywt.Wavelet(wavelet_name)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            - **Name**: {wavelet.name}
            - **Family**: {wavelet.family_name}
            - **Vanishing Moments**: {wavelet.vanishing_moments_psi}
            - **Support Width**: {wavelet.dec_len}
            """)
        with col2:
            st.markdown(f"""
            - **Orthogonal**: {wavelet.orthogonal}
            - **Biorthogonal**: {wavelet.biorthogonal}
            - **Compact Support**: {wavelet.compact_support}
            """)
    except:
        pass

st.markdown("""
### Comparison of Wavelet Families

| Wavelet | Smoothness | Vanishing Moments | Support | Best For |
|---------|-----------|------------------|---------|----------|
| **Haar** | Discontinuous | 1 | 2 | Simple, fast |
| **Daubechies** | Smooth (N≥4) | N | 2N | General purpose |
| **Coiflets** | Smooth | N | 6N-1 | Signal analysis |
| **Biorthogonal** | Smooth | Variable | Variable | Image compression |
| **Symlets** | Smooth | N | 2N | Near symmetric |
| **Morlet** | Very smooth | ∞ | Infinite | Continuous analysis |
| **Mexican Hat** | Smooth | 2 | Infinite | Edge detection |

### Key Observations

1. **Haar**: Simplest, but discontinuous - good for piecewise constant signals
2. **Daubechies**: Most popular, good balance of properties
3. **Coiflets**: Better for signal analysis due to scaling function properties
4. **Biorthogonal**: Used in JPEG 2000, symmetric
5. **Morlet/Mexican Hat**: Continuous wavelets, good for time-frequency analysis

### Example: Signal Decomposition

The bottom-right plot shows how a signal is decomposed into different resolution levels. 
Each level captures different frequency components:
- **High levels**: Low frequency (coarse approximation)
- **Low levels**: High frequency (fine details)
""")

