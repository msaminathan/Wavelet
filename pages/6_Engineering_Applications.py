import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal

st.set_page_config(page_title="Engineering Applications", page_icon="🔧", layout="wide")

# Vibrant styling
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
    
    h2 {
        color: #667eea;
        border-left: 5px solid #f093fb;
        padding-left: 1rem;
        margin-top: 1.5rem;
        background: linear-gradient(90deg, #667eea15, transparent);
        padding: 0.8rem 1rem;
        border-radius: 8px;
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
        🔧 Engineering Applications of Wavelets
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea20, #764ba220);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 2px solid #667eea40;">
    <h2 style="color: #667eea; margin-top: 0; text-align: center;">🌟 Real-World Applications with Code Examples</h2>
    <p style="font-size: 1.1rem; color: #555; text-align: center; margin: 0.5rem 0 0 0;">
        Wavelets are extensively used in engineering. This page demonstrates practical applications with 
        working code examples.
    </p>
</div>
""", unsafe_allow_html=True)

app_tabs = st.tabs([
    "Signal Denoising",
    "Image Compression",
    "Vibration Analysis",
    "ECG Signal Processing",
    "Feature Extraction",
    "Edge Detection"
])

# Application 1: Signal Denoising
with app_tabs[0]:
    st.subheader("1. Signal Denoising")
    st.markdown("""
    **Problem**: Remove noise from a signal while preserving important features.
    
    **Solution**: Use wavelet thresholding - small coefficients (likely noise) are set to zero.
    """)
    
    # Generate noisy signal
    t = np.linspace(0, 1, 1000)
    f1, f2 = 10, 30
    clean_signal = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)
    noise = 0.3 * np.random.randn(len(t))
    noisy_signal = clean_signal + noise
    
    wavelet_denoise = st.selectbox("Wavelet for Denoising", ['db4', 'db8', 'haar', 'coif2'], key='denoise')
    threshold = st.slider("Threshold Multiplier", 0.1, 3.0, 1.0, 0.1, key='thresh')
    
    # Denoise
    coeffs = pywt.wavedec(noisy_signal, wavelet_denoise, level=4)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * threshold * np.sqrt(2 * np.log(len(noisy_signal)))
    coeffs_thresh = [pywt.threshold(c, uthresh, 'soft') for c in coeffs]
    denoised = pywt.waverec(coeffs_thresh, wavelet_denoise)
    if len(denoised) > len(clean_signal):
        denoised = denoised[:len(clean_signal)]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(t, clean_signal, 'b-', linewidth=2, label='Clean Signal')
    axes[0].set_title('Original Clean Signal', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, noisy_signal, 'r-', linewidth=1, alpha=0.7, label='Noisy Signal')
    axes[1].set_title(f'Noisy Signal (SNR: {10*np.log10(np.var(clean_signal)/np.var(noise)):.2f} dB)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, clean_signal, 'b-', linewidth=2, alpha=0.5, label='Original')
    axes[2].plot(t, denoised, 'g-', linewidth=2, label='Denoised')
    axes[2].set_title(f'Denoised Signal (SNR: {10*np.log10(np.var(clean_signal)/np.var(denoised-clean_signal)):.2f} dB)', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.code("""
# Signal Denoising Code
import numpy as np
import pywt

def denoise_signal(signal, wavelet='db4', threshold_mult=1.0):
    # Decompose signal
    coeffs = pywt.wavedec(signal, wavelet, level=4)
    
    # Estimate noise level
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # Calculate threshold (Universal threshold)
    uthresh = sigma * threshold_mult * np.sqrt(2 * np.log(len(signal)))
    
    # Apply soft thresholding
    coeffs_thresh = [pywt.threshold(c, uthresh, 'soft') for c in coeffs]
    
    # Reconstruct
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    
    return denoised[:len(signal)]
    """, language='python')

# Application 2: Image Compression
with app_tabs[1]:
    st.subheader("2. Image Compression")
    st.markdown("""
    **Problem**: Compress images efficiently.
    
    **Solution**: Wavelets provide sparse representation - most coefficients are small and can be discarded.
    """)
    
    # Create a simple test image
    size = 256
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    test_image = np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(size, size)
    
    compression_ratio = st.slider("Compression Ratio (%)", 10, 90, 50, 5, key='comp')
    
    # Compress using wavelets
    coeffs = pywt.wavedec2(test_image, 'bior2.2', level=4)
    coeffs_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)
    
    # Keep only top coefficients
    n_keep = int(len(coeffs_arr.flatten()) * (1 - compression_ratio/100))
    threshold = np.sort(np.abs(coeffs_arr.flatten()))[-n_keep]
    coeffs_arr_comp = coeffs_arr * (np.abs(coeffs_arr) >= threshold)
    coeffs_comp = pywt.array_to_coeffs(coeffs_arr_comp, coeffs_slices, output_format='wavedec2')
    compressed_image = pywt.waverec2(coeffs_comp, 'bior2.2')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(compressed_image, cmap='gray')
    axes[1].set_title(f'Compressed ({compression_ratio}% coefficients removed)', fontweight='bold')
    axes[1].axis('off')
    
    error = np.abs(test_image - compressed_image)
    im = axes[2].imshow(error, cmap='hot')
    axes[2].set_title(f'Error (RMSE: {np.sqrt(np.mean(error**2)):.4f})', fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    st.pyplot(fig)
    
    st.code("""
# Image Compression Code
import numpy as np
import pywt

def compress_image(image, wavelet='bior2.2', compression_ratio=0.5):
    # 2D wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=4)
    coeffs_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)
    
    # Keep top coefficients
    n_keep = int(len(coeffs_arr.flatten()) * (1 - compression_ratio))
    threshold = np.sort(np.abs(coeffs_arr.flatten()))[-n_keep]
    coeffs_arr_comp = coeffs_arr * (np.abs(coeffs_arr) >= threshold)
    
    # Reconstruct
    coeffs_comp = pywt.array_to_coeffs(coeffs_arr_comp, coeffs_slices, output_format='wavedec2')
    compressed = pywt.waverec2(coeffs_comp, wavelet)
    
    return compressed
    """, language='python')

# Application 3: Vibration Analysis
with app_tabs[2]:
    st.subheader("3. Vibration Analysis")
    st.markdown("""
    **Problem**: Analyze mechanical vibrations to detect faults or anomalies.
    
    **Solution**: Wavelets identify frequency components at specific times (e.g., when a bearing fails).
    """)
    
    # Simulate vibration signal with fault
    t_vib = np.linspace(0, 10, 5000)
    # Normal operation: low frequency
    vibration = 0.5 * np.sin(2*np.pi*10*t_vib)
    # Add fault at t=5s: high frequency burst
    fault_time = 5.0
    fault_idx = int(fault_time / t_vib[-1] * len(t_vib))
    vibration[fault_idx:fault_idx+200] += 2.0 * np.sin(2*np.pi*100*t_vib[:200])
    # Add noise
    vibration += 0.1 * np.random.randn(len(vibration))
    
    # Continuous Wavelet Transform
    scales = np.arange(1, 128)
    coeffs_cwt, frequencies = pywt.cwt(vibration, scales, 'morl', 1.0/(t_vib[1]-t_vib[0]))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(t_vib, vibration, 'b-', linewidth=1)
    axes[0].axvline(x=fault_time, color='r', linestyle='--', linewidth=2, label='Fault Occurs')
    axes[0].set_title('Vibration Signal', fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Fourier transform
    freq_vib = np.fft.fftfreq(len(vibration), t_vib[1] - t_vib[0])
    fft_vib = np.abs(np.fft.fft(vibration))
    axes[1].plot(freq_vib[:len(freq_vib)//2], fft_vib[:len(freq_vib)//2], 'r-', linewidth=1.5)
    axes[1].set_title('Fourier Transform (Time info lost)', fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_xlim(0, 120)
    axes[1].grid(True, alpha=0.3)
    
    # CWT scalogram
    coeffs_cwt_abs = np.abs(coeffs_cwt)
    # Ensure 2D array for imshow
    if coeffs_cwt_abs.ndim == 1:
        coeffs_cwt_abs = coeffs_cwt_abs.reshape(1, -1)
    elif coeffs_cwt_abs.ndim == 0:
        coeffs_cwt_abs = np.array([[coeffs_cwt_abs]])
    im = axes[2].imshow(coeffs_cwt_abs, aspect='auto', cmap='jet', 
                        extent=[t_vib[0], t_vib[-1], scales[-1], scales[0]])
    axes[2].axvline(x=fault_time, color='w', linestyle='--', linewidth=2)
    axes[2].set_title('Continuous Wavelet Transform (Time-Frequency)', fontweight='bold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Scale')
    plt.colorbar(im, ax=axes[2], label='Magnitude')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.code("""
# Vibration Analysis Code
import numpy as np
import pywt

def analyze_vibration(signal, sampling_rate):
    # Continuous Wavelet Transform
    scales = np.arange(1, 128)
    coeffs, frequencies = pywt.cwt(signal, scales, 'morl', 1.0/sampling_rate)
    
    # Find anomalies (high energy at specific times)
    energy = np.sum(np.abs(coeffs)**2, axis=0)
    anomaly_threshold = np.mean(energy) + 3 * np.std(energy)
    anomalies = np.where(energy > anomaly_threshold)[0]
    
    return coeffs, anomalies
    """, language='python')

# Application 4: ECG Signal Processing
with app_tabs[3]:
    st.subheader("4. ECG Signal Processing")
    st.markdown("""
    **Problem**: Detect QRS complexes (heartbeats) in ECG signals.
    
    **Solution**: Wavelets can identify sharp transitions (R-peaks) while filtering noise.
    """)
    
    # Simulate ECG signal
    t_ecg = np.linspace(0, 10, 5000)
    ecg_clean = np.zeros_like(t_ecg)
    heart_rate = 72  # bpm
    beat_interval = 60.0 / heart_rate
    
    for i in range(int(10 / beat_interval)):
        beat_time = i * beat_interval
        beat_idx = int(beat_time / t_ecg[-1] * len(t_ecg))
        if beat_idx < len(t_ecg) - 100:
            # QRS complex (simplified)
            t_beat = np.linspace(0, 0.1, 100)
            qrs = np.exp(-((t_beat - 0.05)**2) / 0.001) * 2.0
            ecg_clean[beat_idx:beat_idx+len(qrs)] += qrs
    
    ecg_noisy = ecg_clean + 0.2 * np.random.randn(len(ecg_clean))
    
    # Detect R-peaks using wavelets
    coeffs_ecg = pywt.wavedec(ecg_noisy, 'db4', level=6)
    # R-peaks appear in detail coefficients at level 2-3
    detail = coeffs_ecg[3]  # Level 3 detail
    peaks, _ = signal.find_peaks(np.abs(detail), height=np.std(detail)*2, distance=100)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(t_ecg, ecg_clean, 'b-', linewidth=1.5, label='Clean ECG')
    axes[0].set_title('Clean ECG Signal', fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_ecg, ecg_noisy, 'r-', linewidth=1, alpha=0.7, label='Noisy ECG')
    axes[1].set_title('Noisy ECG Signal', fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Map peaks back to original signal
    peak_times = peaks * (t_ecg[-1] / len(detail))
    axes[2].plot(t_ecg, ecg_noisy, 'k-', linewidth=1, alpha=0.5, label='ECG')
    axes[2].plot(peak_times, ecg_noisy[peaks * len(t_ecg) // len(detail)], 
                 'ro', markersize=10, label='Detected R-peaks')
    axes[2].set_title(f'R-Peak Detection ({len(peaks)} beats detected)', fontweight='bold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.code("""
# ECG QRS Detection Code
import numpy as np
import pywt
from scipy import signal

def detect_qrs_complexes(ecg_signal, wavelet='db4', level=6):
    # Wavelet decomposition
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
    
    # R-peaks appear in detail coefficients (levels 2-3)
    detail = coeffs[3]  # Adjust level as needed
    
    # Find peaks
    peaks, properties = signal.find_peaks(
        np.abs(detail), 
        height=np.std(detail) * 2,
        distance=100  # Minimum distance between peaks
    )
    
    # Calculate heart rate
    sampling_rate = 500  # Hz
    rr_intervals = np.diff(peaks) / sampling_rate
    heart_rate = 60.0 / np.mean(rr_intervals)
    
    return peaks, heart_rate
    """, language='python')

# Application 5: Feature Extraction
with app_tabs[4]:
    st.subheader("5. Feature Extraction")
    st.markdown("""
    **Problem**: Extract meaningful features from signals for machine learning.
    
    **Solution**: Wavelet coefficients provide compact, informative features.
    """)
    
    # Generate different signal classes
    t_feat = np.linspace(0, 1, 1000)
    signals = {
        'Class A': np.sin(2*np.pi*10*t_feat),
        'Class B': np.sin(2*np.pi*30*t_feat),
        'Class C': np.sin(2*np.pi*10*t_feat) + 0.5*np.sin(2*np.pi*30*t_feat)
    }
    
    # Extract features
    features = {}
    for name, sig in signals.items():
        coeffs = pywt.wavedec(sig, 'db4', level=4)
        # Use energy in each level as features
        features[name] = [np.sum(c**2) for c in coeffs]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot signals
    for i, (name, sig) in enumerate(signals.items()):
        axes[0, 0].plot(t_feat, sig, linewidth=1.5, label=name)
    axes[0, 0].set_title('Different Signal Classes', fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot features
    feature_matrix = np.array(list(features.values()))
    # Ensure 2D array for imshow
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(1, -1)
    elif feature_matrix.ndim == 0:
        feature_matrix = np.array([[feature_matrix]])
    im = axes[0, 1].imshow(feature_matrix, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('Wavelet Energy Features', fontweight='bold')
    axes[0, 1].set_xticks(range(len(feature_matrix[0])))
    axes[0, 1].set_xticklabels([f'Level {i}' for i in range(len(feature_matrix[0]))])
    axes[0, 1].set_yticks(range(len(signals)))
    axes[0, 1].set_yticklabels(list(signals.keys()))
    plt.colorbar(im, ax=axes[0, 1])
    
    # Bar plot
    x_pos = np.arange(len(feature_matrix[0]))
    width = 0.25
    for i, (name, feat) in enumerate(features.items()):
        axes[1, 0].bar(x_pos + i*width, feat, width, label=name)
    axes[1, 0].set_title('Feature Comparison', fontweight='bold')
    axes[1, 0].set_xlabel('Wavelet Level')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_xticks(x_pos + width)
    axes[1, 0].set_xticklabels([f'L{i}' for i in range(len(feature_matrix[0]))])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 2D feature space
    axes[1, 1].scatter([f[0] for f in features.values()], 
                       [f[1] for f in features.values()], 
                       s=200, alpha=0.7)
    for i, name in enumerate(signals.keys()):
        axes[1, 1].annotate(name, 
                           (features[name][0], features[name][1]),
                           fontsize=12, fontweight='bold')
    axes[1, 1].set_title('2D Feature Space (Level 0 vs Level 1)', fontweight='bold')
    axes[1, 1].set_xlabel('Energy at Level 0')
    axes[1, 1].set_ylabel('Energy at Level 1')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.code("""
# Feature Extraction Code
import numpy as np
import pywt

def extract_wavelet_features(signal, wavelet='db4', level=4):
    # Decompose signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Extract features
    features = {
        'energy': [np.sum(c**2) for c in coeffs],
        'mean': [np.mean(np.abs(c)) for c in coeffs],
        'std': [np.std(c) for c in coeffs],
        'max': [np.max(np.abs(c)) for c in coeffs]
    }
    
    # Flatten for ML
    feature_vector = np.concatenate([
        features['energy'],
        features['mean'],
        features['std'],
        features['max']
    ])
    
    return feature_vector
    """, language='python')

# Application 6: Edge Detection
with app_tabs[5]:
    st.subheader("6. Edge Detection in Images")
    st.markdown("""
    **Problem**: Detect edges and boundaries in images.
    
    **Solution**: Wavelets are excellent at detecting discontinuities (edges).
    """)
    
    # Create test image with edges
    size_edge = 256
    test_img = np.zeros((size_edge, size_edge))
    test_img[50:150, 50:150] = 1.0  # Square
    test_img[100:200, 100:200] = 0.5  # Overlapping square
    test_img += 0.1 * np.random.randn(size_edge, size_edge)  # Noise
    
    # Edge detection using wavelets
    coeffs_edge = pywt.wavedec2(test_img, 'haar', level=3)
    # Horizontal and vertical details contain edge information
    _, (LH, HL, HH) = coeffs_edge[1]  # Level 1 details
    
    # Combine horizontal and vertical edges
    edges = np.sqrt(LH**2 + HL**2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(LH, cmap='gray')
    axes[0, 1].set_title('Horizontal Edges (LH)', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(HL, cmap='gray')
    axes[1, 0].set_title('Vertical Edges (HL)', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edges, cmap='hot')
    axes[1, 1].set_title('Combined Edges', fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.code("""
# Edge Detection Code
import numpy as np
import pywt

def detect_edges_wavelet(image, wavelet='haar', level=3):
    # 2D wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Get horizontal and vertical details
    _, (LH, HL, HH) = coeffs[1]  # Level 1
    
    # Combine for edge map
    edge_map = np.sqrt(LH**2 + HL**2)
    
    # Threshold
    threshold = np.mean(edge_map) + 2 * np.std(edge_map)
    edges = edge_map > threshold
    
    return edges, edge_map
    """, language='python')

st.markdown("""
---

## Summary of Engineering Applications

| Application | Wavelet Used | Key Benefit |
|------------|--------------|-------------|
| **Signal Denoising** | db4, db8 | Preserves edges while removing noise |
| **Image Compression** | bior2.2, bior4.4 | Sparse representation, JPEG 2000 |
| **Vibration Analysis** | Morlet, db4 | Time-frequency localization |
| **ECG Processing** | db4, db6 | QRS complex detection |
| **Feature Extraction** | db4, coif2 | Compact, informative features |
| **Edge Detection** | Haar, db2 | Discontinuity detection |

### Why Wavelets Work Well

1. **Localization**: Capture local features in time/space
2. **Multi-resolution**: Analyze at multiple scales
3. **Sparsity**: Few coefficients needed for representation
4. **Adaptability**: Different wavelets for different applications
""")

