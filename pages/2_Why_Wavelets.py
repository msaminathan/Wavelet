import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from shared_navigation import render_navigation

st.set_page_config(page_title="Why Wavelets?", page_icon="❓", layout="wide")

# Render shared navigation
render_navigation()

st.title("❓ Why Wavelets? Understanding the Need")

st.markdown("""
## The Problem with Traditional Methods

### Limitations of Fourier Transform

The **Fourier Transform** is excellent for analyzing periodic signals and determining frequency content, 
but it has a critical limitation: **it loses all time information**.

#### Example: Two Different Signals, Same Fourier Transform
""")

# Create example showing Fourier Transform limitation
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Signal 1: High frequency at beginning
t = np.linspace(0, 1, 1000)
signal1 = np.zeros_like(t)
signal1[:250] = np.sin(2 * np.pi * 50 * t[:250])  # High freq at start
signal1[250:] = np.sin(2 * np.pi * 5 * t[250:])   # Low freq at end

# Signal 2: Low frequency at beginning
signal2 = np.zeros_like(t)
signal2[:250] = np.sin(2 * np.pi * 5 * t[:250])   # Low freq at start
signal2[250:] = np.sin(2 * np.pi * 50 * t[250:])  # High freq at end

# Plot signals
axes[0, 0].plot(t, signal1, 'b-', linewidth=1.5)
axes[0, 0].set_title('Signal 1: High→Low Frequency', fontweight='bold')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, signal2, 'r-', linewidth=1.5)
axes[0, 1].set_title('Signal 2: Low→High Frequency', fontweight='bold')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].grid(True, alpha=0.3)

# Fourier transforms (magnitude)
freq = np.fft.fftfreq(len(t), t[1] - t[0])
fft1 = np.abs(np.fft.fft(signal1))
fft2 = np.abs(np.fft.fft(signal2))

axes[1, 0].plot(freq[:len(freq)//2], fft1[:len(freq)//2], 'b-', linewidth=1.5)
axes[1, 0].set_title('Fourier Transform of Signal 1', fontweight='bold')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Magnitude')
axes[1, 0].set_xlim(0, 60)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(freq[:len(freq)//2], fft2[:len(freq)//2], 'r-', linewidth=1.5)
axes[1, 1].set_title('Fourier Transform of Signal 2', fontweight='bold')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_ylabel('Magnitude')
axes[1, 1].set_xlim(0, 60)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**Notice**: Both signals have nearly identical Fourier transforms! The Fourier transform tells us 
*what* frequencies are present, but not *when* they occur. This is a fundamental limitation.

### The Solution: Wavelet Transform

Wavelets solve this problem by providing **time-frequency localization**:

1. **Short-Time Fourier Transform (STFT)** - A compromise, but with fixed window size
2. **Wavelet Transform** - Adaptive window size (better for multi-scale signals)

### Key Advantages of Wavelets

#### 1. **Time-Frequency Localization**
- Know both *what* frequency and *when* it occurs
- Perfect for non-stationary signals (signals that change over time)

#### 2. **Multi-Resolution Analysis**
- Analyze signals at different scales simultaneously
- Capture both fine details and broad trends

#### 3. **Sparse Representation**
- Many signals can be represented with few wavelet coefficients
- Enables efficient compression

#### 4. **Edge Detection**
- Excellent at detecting discontinuities and sharp changes
- Better than Fourier methods for signals with edges

### Real-World Scenarios Where Wavelets Excel

#### Scenario 1: Seismic Signal Analysis
""")

# Seismic-like signal example
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Create a seismic-like signal with events at different times
t_seismic = np.linspace(0, 10, 5000)
seismic = np.random.randn(len(t_seismic)) * 0.1  # Background noise
seismic[1000:1100] += np.sin(2 * np.pi * 20 * np.linspace(0, 0.1, 100))  # Event 1
seismic[2500:2550] += np.sin(2 * np.pi * 5 * np.linspace(0, 0.05, 50))   # Event 2
seismic[4000:4100] += np.sin(2 * np.pi * 30 * np.linspace(0, 0.1, 100))  # Event 3

axes[0].plot(t_seismic, seismic, 'b-', linewidth=0.5)
axes[0].set_title('Seismic Signal with Events at Different Times', fontweight='bold')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=2, color='r', linestyle='--', alpha=0.5, label='Event 1')
axes[0].axvline(x=5, color='g', linestyle='--', alpha=0.5, label='Event 2')
axes[0].axvline(x=8, color='m', linestyle='--', alpha=0.5, label='Event 3')
axes[0].legend()

# Show why Fourier fails
freq_seismic = np.fft.fftfreq(len(t_seismic), t_seismic[1] - t_seismic[0])
fft_seismic = np.abs(np.fft.fft(seismic))

axes[1].plot(freq_seismic[:len(freq_seismic)//2], fft_seismic[:len(freq_seismic)//2], 'r-', linewidth=1.5)
axes[1].set_title('Fourier Transform (Time Info Lost)', fontweight='bold')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude')
axes[1].set_xlim(0, 40)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**Problem**: Fourier transform shows frequencies but not when events occurred.
**Solution**: Wavelet transform shows both frequency content AND timing of each event.

#### Scenario 2: Image Compression

Wavelets excel at image compression because:
- Natural images have smooth regions (few coefficients needed)
- Edges are localized (captured efficiently by wavelets)
- **JPEG 2000** uses wavelets instead of DCT (used in JPEG)

#### Scenario 3: Signal Denoising

Wavelets can separate signal from noise because:
- Signals often have structure (few large coefficients)
- Noise is random (many small coefficients)
- Threshold small coefficients → remove noise

### When to Use Wavelets?

**Use Wavelets When:**
- ✅ Signal is non-stationary (changes over time)
- ✅ You need time-frequency information
- ✅ Signal has features at multiple scales
- ✅ Compression is important
- ✅ Edge detection is needed
- ✅ Signal has localized features

**Use Fourier When:**
- ✅ Signal is stationary (doesn't change over time)
- ✅ Only frequency information is needed
- ✅ Signal is periodic
- ✅ You need exact frequency resolution

### Code Example: Comparing Fourier vs Wavelet

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

# Create a signal with frequency that changes over time
t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * 10 * t) * (t < 0.5) + np.sin(2 * np.pi * 50 * t) * (t >= 0.5)

# Fourier Transform
X_fft = np.fft.fft(x)
freq = np.fft.fftfreq(len(t), t[1] - t[0])

# Wavelet Transform (Continuous Wavelet Transform)
scales = np.arange(1, 128)
coeffs, frequencies = pywt.cwt(x, scales, 'morl', 1.0/len(t))

# Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(t, x)
axes[0].set_title('Original Signal')
axes[1].plot(freq[:len(freq)//2], np.abs(X_fft[:len(freq)//2]))
axes[1].set_title('Fourier Transform (no time info)')
coeffs_abs = np.abs(coeffs)
# Ensure 2D array for imshow
if coeffs_abs.ndim == 1:
    coeffs_abs = coeffs_abs.reshape(1, -1)
elif coeffs_abs.ndim == 0:
    coeffs_abs = np.array([[coeffs_abs]])
axes[2].imshow(coeffs_abs, aspect='auto', cmap='jet')
axes[2].set_title('Wavelet Transform (time-frequency)')
plt.tight_layout()
plt.show()
```

### Summary

Wavelets are needed because:
1. **Real signals are non-stationary** - They change over time
2. **Time information matters** - We need to know when events occur
3. **Multi-scale analysis** - Signals have features at different scales
4. **Efficient representation** - Better compression and denoising
5. **Edge detection** - Better than Fourier for discontinuous signals

The next section will dive deep into the mathematical foundations that make wavelets work!
""")


