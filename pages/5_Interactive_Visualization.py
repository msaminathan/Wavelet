import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal

st.set_page_config(page_title="Interactive Visualization", page_icon="🎮", layout="wide")

# Vibrant styling for interactive page
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
    
    .stSlider > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(90deg, #f093fb, #4facfe);
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
        🎮 Interactive Wavelet Visualization
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea20, #764ba220);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 2px solid #667eea40;
            text-align: center;">
    <h2 style="color: #667eea; margin-top: 0;">✨ Explore Wavelets Interactively</h2>
    <p style="font-size: 1.1rem; color: #555; margin: 0;">
        Use the controls below to create custom signals and visualize their wavelet transforms.
    </p>
</div>
""", unsafe_allow_html=True)

# Vibrant sidebar header
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;">
    <h3 style="color: white; margin: 0; font-size: 1.3rem;">⚙️ Signal Parameters</h3>
</div>
""", unsafe_allow_html=True)

signal_type = st.sidebar.selectbox(
    "Signal Type",
    ["Chirp (Frequency Sweep)", "Two Tones", "Impulse", "Noisy Signal", "Custom"]
)

# Common parameters
duration = st.sidebar.slider("Duration (seconds)", 0.1, 5.0, 1.0, 0.1)
sampling_rate = st.sidebar.slider("Sampling Rate (Hz)", 100, 10000, 1000, 100)
n_samples = int(duration * sampling_rate)
t = np.linspace(0, duration, n_samples)

# Generate signal based on type
if signal_type == "Chirp (Frequency Sweep)":
    f_start = st.sidebar.slider("Start Frequency (Hz)", 1, 100, 5)
    f_end = st.sidebar.slider("End Frequency (Hz)", 1, 200, 50)
    x = signal.chirp(t, f_start, duration, f_end, method='linear')
    
elif signal_type == "Two Tones":
    f1 = st.sidebar.slider("Frequency 1 (Hz)", 1, 100, 10)
    f2 = st.sidebar.slider("Frequency 2 (Hz)", 1, 100, 30)
    t_switch = st.sidebar.slider("Switch Time (fraction)", 0.1, 0.9, 0.5, 0.1)
    switch_idx = int(t_switch * len(t))
    x = np.zeros_like(t)
    x[:switch_idx] = np.sin(2 * np.pi * f1 * t[:switch_idx])
    x[switch_idx:] = np.sin(2 * np.pi * f2 * t[switch_idx:])
    
elif signal_type == "Impulse":
    impulse_time = st.sidebar.slider("Impulse Time (fraction)", 0.0, 1.0, 0.5, 0.01)
    impulse_idx = int(impulse_time * len(t))
    x = np.zeros_like(t)
    x[impulse_idx] = 1.0
    
elif signal_type == "Noisy Signal":
    f_signal = st.sidebar.slider("Signal Frequency (Hz)", 1, 100, 20)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.3, 0.1)
    x = np.sin(2 * np.pi * f_signal * t) + noise_level * np.random.randn(len(t))
    
else:  # Custom
    st.sidebar.markdown("### Custom Signal")
    custom_expr = st.sidebar.text_input("Expression (use 't' for time)", "np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*30*t)")
    try:
        x = eval(custom_expr)
        if not isinstance(x, np.ndarray):
            x = np.array([x] * len(t))
    except:
        x = np.sin(2 * np.pi * 10 * t)

# Vibrant sidebar header for wavelet parameters
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #f093fb, #4facfe);
            padding: 1rem;
            border-radius: 10px;
            margin: 1.5rem 0 1rem 0;
            text-align: center;">
    <h3 style="color: white; margin: 0; font-size: 1.3rem;">🌊 Wavelet Parameters</h3>
</div>
""", unsafe_allow_html=True)
wavelet_name = st.sidebar.selectbox(
    "Wavelet",
    ['haar', 'db4', 'db8', 'coif2', 'bior2.2', 'sym4']
)

max_level = st.sidebar.slider("Decomposition Level", 1, 8, 4)

# Perform wavelet transform
try:
    coeffs = pywt.wavedec(x, wavelet_name, level=max_level)
    coeffs_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)
except:
    st.error("Error performing wavelet transform")
    st.stop()

# Create visualizations
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Original signal
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, x, 'b-', linewidth=1.5)
ax1.set_title('Original Signal', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.grid(True, alpha=0.3)

# Fourier transform
ax2 = fig.add_subplot(gs[1, 0])
freq = np.fft.fftfreq(len(x), t[1] - t[0])
fft_x = np.fft.fft(x)
ax2.plot(freq[:len(freq)//2], np.abs(fft_x[:len(freq)//2]), 'r-', linewidth=1.5)
ax2.set_title('Fourier Transform (Magnitude)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.grid(True, alpha=0.3)

# Wavelet coefficients
ax3 = fig.add_subplot(gs[1, 1])
coeffs_arr_abs = np.abs(coeffs_arr)
# Ensure 2D array for imshow
if coeffs_arr_abs.ndim == 1:
    # Reshape to 2D if 1D
    coeffs_arr_abs = coeffs_arr_abs.reshape(1, -1)
elif coeffs_arr_abs.ndim == 0:
    # If scalar, create a 1x1 array
    coeffs_arr_abs = np.array([[coeffs_arr_abs]])
im = ax3.imshow(coeffs_arr_abs, aspect='auto', cmap='jet', interpolation='nearest')
ax3.set_title('Wavelet Coefficients (Magnitude)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Time Index')
ax3.set_ylabel('Scale Level')
plt.colorbar(im, ax=ax3)

# Wavelet decomposition
ax4 = fig.add_subplot(gs[2, :])
y_pos = 0
colors = plt.cm.viridis(np.linspace(0, 1, len(coeffs)))
for i, (coeff, color) in enumerate(zip(coeffs, colors)):
    level = len(coeffs) - 1 - i
    t_coeff = np.linspace(0, duration, len(coeff))
    offset = y_pos
    ax4.plot(t_coeff, coeff + offset, color=color, linewidth=1.5, label=f'Level {level}')
    y_pos += np.max(np.abs(coeff)) * 2.5
ax4.set_title('Wavelet Decomposition by Level', fontsize=12, fontweight='bold')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Amplitude (offset)')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# Reconstruction
ax5 = fig.add_subplot(gs[3, 0])
x_recon = pywt.waverec(coeffs, wavelet_name)
if len(x_recon) > len(x):
    x_recon = x_recon[:len(x)]
elif len(x_recon) < len(x):
    x_recon = np.pad(x_recon, (0, len(x) - len(x_recon)))
ax5.plot(t, x, 'b-', linewidth=1.5, alpha=0.7, label='Original')
ax5.plot(t, x_recon, 'r--', linewidth=1.5, label='Reconstructed')
ax5.set_title('Original vs Reconstructed', fontsize=12, fontweight='bold')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Amplitude')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Error
ax6 = fig.add_subplot(gs[3, 1])
error = x - x_recon
ax6.plot(t, error, 'g-', linewidth=1.5)
ax6.set_title(f'Reconstruction Error (RMSE: {np.sqrt(np.mean(error**2)):.6f})', 
              fontsize=12, fontweight='bold')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Error')
ax6.grid(True, alpha=0.3)

st.pyplot(fig)

# Statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Signal Length", len(x))
with col2:
    st.metric("Decomposition Levels", len(coeffs))
with col3:
    st.metric("Reconstruction RMSE", f"{np.sqrt(np.mean(error**2)):.6f}")
with col4:
    energy_original = np.sum(x**2)
    energy_coeffs = np.sum([np.sum(c**2) for c in coeffs])
    st.metric("Energy Conservation", f"{(energy_coeffs/energy_original*100):.2f}%")

# Denoising example with vibrant header
st.markdown("""
<div style="background: linear-gradient(135deg, #fee14020, #fa709a20);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 2rem 0 1rem 0;
            border-left: 5px solid #fee140;">
    <h2 style="color: #fa709a; margin: 0;">🧹 Wavelet Denoising Example</h2>
    <p style="color: #666; margin: 0.5rem 0 0 0;">See how wavelets can remove noise while preserving signal features</p>
</div>
""", unsafe_allow_html=True)

threshold_type = st.selectbox("Threshold Type", ["soft", "hard"])
threshold_mode = st.selectbox("Threshold Mode", ["symmetric", "antisymmetric", "soft", "hard", "garrote", "greater", "less"])

# Add noise
noise_level = st.slider("Noise Level", 0.0, 1.0, 0.2, 0.05)
x_noisy = x + noise_level * np.random.randn(len(x))

# Denoise
try:
    coeffs_noisy = pywt.wavedec(x_noisy, wavelet_name, level=max_level)
    sigma = np.median(np.abs(coeffs_noisy[-1])) / 0.6745  # Estimate noise level
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs_thresh = [pywt.threshold(c, uthresh, threshold_type) for c in coeffs_noisy]
    x_denoised = pywt.waverec(coeffs_thresh, wavelet_name)
    if len(x_denoised) > len(x):
        x_denoised = x_denoised[:len(x)]
    elif len(x_denoised) < len(x):
        x_denoised = np.pad(x_denoised, (0, len(x) - len(x_denoised)))
except Exception as e:
    st.error(f"Error in denoising: {e}")
    x_denoised = x_noisy

fig2, axes = plt.subplots(3, 1, figsize=(14, 8))
axes[0].plot(t, x, 'b-', linewidth=1.5, label='Original')
axes[0].set_title('Original Signal', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, x_noisy, 'r-', linewidth=1, alpha=0.7, label='Noisy')
axes[1].set_title(f'Noisy Signal (SNR: {10*np.log10(np.var(x)/np.var(x_noisy-x)):.2f} dB)', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, x, 'b-', linewidth=1.5, alpha=0.5, label='Original')
axes[2].plot(t, x_denoised, 'g-', linewidth=1.5, label='Denoised')
axes[2].set_title(f'Denoised Signal (SNR: {10*np.log10(np.var(x)/np.var(x_denoised-x)):.2f} dB)', fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

plt.tight_layout()
st.pyplot(fig2)

st.markdown("""
### Code for Interactive Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal

# Generate signal
t = np.linspace(0, 1, 1000)
x = signal.chirp(t, 5, 1, 50, method='linear')

# Wavelet decomposition
wavelet = 'db4'
level = 4
coeffs = pywt.wavedec(x, wavelet, level=level)

# Visualize coefficients
coeffs_arr, _ = pywt.coeffs_to_array(coeffs)
coeffs_arr_abs = np.abs(coeffs_arr)
# Ensure 2D array for imshow
if coeffs_arr_abs.ndim == 1:
    coeffs_arr_abs = coeffs_arr_abs.reshape(1, -1)
plt.imshow(coeffs_arr_abs, aspect='auto', cmap='jet')
plt.title('Wavelet Coefficients')
plt.xlabel('Time Index')
plt.ylabel('Scale Level')
plt.colorbar()
plt.show()

# Reconstruction
x_recon = pywt.waverec(coeffs, wavelet)
```
""")

