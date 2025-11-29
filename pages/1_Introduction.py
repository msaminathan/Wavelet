import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from shared_navigation import render_navigation

st.set_page_config(page_title="Introduction", page_icon="📚", layout="wide")

# Render shared navigation
render_navigation()

# Vibrant page styling
st.markdown("""
<style>
    h1 {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #667eea;
        border-left: 5px solid #f093fb;
        padding-left: 1rem;
        margin-top: 2rem;
        background: linear-gradient(90deg, #667eea15, transparent);
        padding: 0.8rem 1rem;
        border-radius: 8px;
    }
    
    h3 {
        color: #764ba2;
        margin-top: 1.5rem;
    }
    
    .stMarkdown ul li {
        background: linear-gradient(90deg, #667eea10, transparent);
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 style="background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;
               font-size: 2.8rem;
               font-weight: 800;
               margin: 0;">
        📚 Introduction to Wavelets
    </h1>
</div>
""", unsafe_allow_html=True)

st.html("""
<div style="background: linear-gradient(135deg, #667eea20, #764ba220);
            padding: 2rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            border: 2px solid #667eea40;">
    <h2 style="color: #667eea; margin-top: 0;">🌊 What are Wavelets?</h2>
    <p style="font-size: 1.15rem; color: #555; line-height: 1.8;">
        <strong style="color: #764ba2; font-size: 1.2rem;">Wavelets</strong> are mathematical functions that provide a way to analyze signals in both time and frequency domains simultaneously. 
        The term "wavelet" comes from the fact that these functions are small waves (oscillations) that are localized in time.
    </p>
</div>
""")

st.html("""
<div style="background: linear-gradient(135deg, #f093fb20, #4facfe20);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            border-left: 5px solid #f093fb;">
    <h3 style="color: #764ba2; margin-top: 0;">✨ Key Characteristics</h3>
    
    <div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.8rem 0; border-left: 4px solid #667eea;">
        <strong style="color: #667eea; font-size: 1.1rem;">1. Time-Frequency Localization</strong>
        <p style="color: #666; margin: 0.5rem 0 0 0;">
            Unlike Fourier transforms, wavelets can identify both <em>when</em> and <em>at what frequency</em> 
            events occur in a signal.
        </p>
    </div>
    
    <div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.8rem 0; border-left: 4px solid #764ba2;">
        <strong style="color: #764ba2; font-size: 1.1rem;">2. Multi-Resolution Analysis</strong>
        <p style="color: #666; margin: 0.5rem 0 0 0;">
            Wavelets allow us to analyze signals at different scales (resolutions), making them 
            ideal for signals with features at multiple scales.
        </p>
    </div>
    
    <div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.8rem 0; border-left: 4px solid #f093fb;">
        <strong style="color: #f093fb; font-size: 1.1rem;">3. Compact Support</strong>
        <p style="color: #666; margin: 0.5rem 0 0 0;">
            Many wavelets have finite duration, meaning they are zero outside a finite interval.
        </p>
    </div>
</div>
""")

st.html("""
<div style="background: linear-gradient(135deg, #fee14020, #fa709a20);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            border-left: 5px solid #fee140;">
    <h3 style="color: #fa709a; margin-top: 0;">📅 Historical Context</h3>
    
    <div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0; padding: 1rem; background: white; border-radius: 10px;">
        <span style="font-size: 2rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🕰️</span>
        <div>
            <strong style="color: #667eea;">1980s</strong>
            <p style="color: #666; margin: 0.3rem 0 0 0;">Wavelet theory was developed by mathematicians like Jean Morlet, Yves Meyer, and Ingrid Daubechies</p>
        </div>
    </div>
    
    <div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0; padding: 1rem; background: white; border-radius: 10px;">
        <span style="font-size: 2rem; background: linear-gradient(135deg, #f093fb, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📈</span>
        <div>
            <strong style="color: #764ba2;">1990s</strong>
            <p style="color: #666; margin: 0.3rem 0 0 0;">Rapid adoption in signal processing, image compression, and numerical analysis</p>
        </div>
    </div>
    
    <div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0; padding: 1rem; background: white; border-radius: 10px;">
        <span style="font-size: 2rem; background: linear-gradient(135deg, #4facfe, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🚀</span>
        <div>
            <strong style="color: #4facfe;">Today</strong>
            <p style="color: #666; margin: 0.3rem 0 0 0;">Used extensively in data compression (JPEG 2000), denoising, feature extraction, and more</p>
        </div>
    </div>
</div>
""")

st.markdown("""
### Why "Wavelet" Instead of "Wave"?

A **wave** (like a sine wave) extends infinitely in time, while a **wavelet** is a "small wave" that:
- Starts at zero
- Oscillates for a short period
- Returns to zero

This finite duration makes wavelets perfect for analyzing localized features in signals.

### Basic Wavelet Properties

A wavelet function ψ(t) must satisfy:

1. **Admissibility Condition**: ∫ ψ(t) dt = 0 (zero mean)
2. **Energy Normalization**: ∫ |ψ(t)|² dt = 1
3. **Decay**: The function should decay rapidly to zero

### Types of Wavelets

- **Orthogonal Wavelets**: Daubechies, Haar, Coiflets
- **Biorthogonal Wavelets**: Used in image compression
- **Continuous Wavelets**: Morlet, Mexican Hat
- **Discrete Wavelets**: Used in practical applications

### Simple Example: Haar Wavelet

The simplest wavelet is the **Haar wavelet**, which looks like a step function:
""")

# Generate Haar wavelet visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

t = np.linspace(-1, 1, 1000)
haar = np.zeros_like(t)
haar[(t >= 0) & (t < 0.5)] = 1
haar[(t >= 0.5) & (t < 1)] = -1

axes[0].plot(t, haar, 'b-', linewidth=2)
axes[0].set_title('Haar Wavelet', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Show scaling function (father wavelet)
scaling = np.zeros_like(t)
scaling[(t >= 0) & (t < 1)] = 1

axes[1].plot(t, scaling, 'r-', linewidth=2)
axes[1].set_title('Haar Scaling Function', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
### Code to Generate Haar Wavelet

```python
import numpy as np
import matplotlib.pyplot as plt

# Define time axis
t = np.linspace(-1, 1, 1000)

# Create Haar wavelet
haar = np.zeros_like(t)
haar[(t >= 0) & (t < 0.5)] = 1      # Positive step
haar[(t >= 0.5) & (t < 1)] = -1     # Negative step

# Plot
plt.figure(figsize=(8, 4))
plt.plot(t, haar, 'b-', linewidth=2)
plt.title('Haar Wavelet')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```

### Wavelet Transform vs Fourier Transform

| Feature | Fourier Transform | Wavelet Transform |
|---------|------------------|-------------------|
| **Time Information** | Lost | Preserved |
| **Frequency Information** | Exact | Approximate |
| **Best For** | Stationary signals | Non-stationary signals |
| **Resolution** | Fixed | Adaptive (multi-resolution) |
| **Basis Functions** | Sine/Cosine (infinite) | Wavelets (localized) |
""")

st.html("""
<div style="background: linear-gradient(135deg, #30cfd020, #33086720);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            border-left: 5px solid #30cfd0;">
    <h3 style="color: #330867; margin-top: 0;">🎯 Applications Overview</h3>
    <p style="color: #555; margin-bottom: 1rem;">Wavelets are used in:</p>
    
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
        <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea;">
            <strong style="color: #667eea;">📊 Signal Processing</strong>
            <p style="color: #666; margin: 0.3rem 0 0 0; font-size: 0.9rem;">Noise reduction, feature extraction</p>
        </div>
        <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #764ba2;">
            <strong style="color: #764ba2;">🖼️ Image Processing</strong>
            <p style="color: #666; margin: 0.3rem 0 0 0; font-size: 0.9rem;">Compression, denoising, edge detection</p>
        </div>
        <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #f093fb;">
            <strong style="color: #f093fb;">🔊 Audio Processing</strong>
            <p style="color: #666; margin: 0.3rem 0 0 0; font-size: 0.9rem;">Compression, analysis</p>
        </div>
        <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #4facfe;">
            <strong style="color: #4facfe;">📈 Data Analysis</strong>
            <p style="color: #666; margin: 0.3rem 0 0 0; font-size: 0.9rem;">Time series analysis, anomaly detection</p>
        </div>
        <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #00f2fe;">
            <strong style="color: #00f2fe;">🔬 Scientific Computing</strong>
            <p style="color: #666; margin: 0.3rem 0 0 0; font-size: 0.9rem;">Numerical analysis, solving PDEs</p>
        </div>
        <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #f5576c;">
            <strong style="color: #f5576c;">🏗️ Engineering</strong>
            <p style="color: #666; margin: 0.3rem 0 0 0; font-size: 0.9rem;">Vibration analysis, structural health monitoring</p>
        </div>
    </div>
</div>
""")

st.markdown("""
### Next Steps

Continue to the next pages to learn:
1. **Why Wavelets?** - Understanding the motivation
2. **Mathematical Foundation** - Deep theoretical background
3. **Examples** - Visual demonstrations
4. **Applications** - Real-world engineering uses
""")

