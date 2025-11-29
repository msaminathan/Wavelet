import streamlit as st
from shared_navigation import render_navigation

st.set_page_config(page_title="Additional Thoughts", page_icon="💭", layout="wide")

# Render shared navigation
render_navigation()

st.title("💭 Additional Thoughts & What You Might Have Missed")

st.markdown("""
## Important Considerations and Advanced Topics

This page covers additional important aspects of wavelets that complement the main content.
""")

topics = st.tabs([
    "Computational Complexity",
    "Wavelet Selection Guide",
    "Limitations & Trade-offs",
    "Advanced Topics",
    "Practical Tips",
    "Common Mistakes"
])

with topics[0]:
    st.subheader("Computational Complexity")
    st.markdown("""
    ### Time Complexity
    
    **Discrete Wavelet Transform (DWT)**:
    - **Forward Transform**: O(N) where N is signal length
    - **Inverse Transform**: O(N)
    - **Much faster than FFT** for many operations (FFT is O(N log N))
    
    **Why DWT is Fast**:
    - Uses filter banks (convolution operations)
    - Downsampling reduces computation at each level
    - No need for complex exponentials
    
    **Continuous Wavelet Transform (CWT)**:
    - **Time Complexity**: O(N × M) where M is number of scales
    - **Slower than DWT** but provides more detailed time-frequency information
    - Often used for analysis, DWT for processing
    
    ### Space Complexity
    
    - **DWT**: O(N) - same as input (for orthogonal wavelets)
    - **CWT**: O(N × M) - scales × time samples
    - **2D DWT**: O(N²) for N×N image
    
    ### Practical Considerations
    
    ```python
    # For large signals, consider:
    # 1. Processing in chunks
    # 2. Using faster wavelets (haar, db2)
    # 3. Limiting decomposition levels
    # 4. Using DWT instead of CWT when possible
    ```
    """)

with topics[1]:
    st.subheader("Wavelet Selection Guide")
    st.markdown("""
    ### Decision Tree
    
    **Choose Haar if**:
    - ✅ Speed is critical
    - ✅ Signal is piecewise constant
    - ✅ Simple implementation needed
    - ❌ Don't need smooth reconstruction
    
    **Choose Daubechies (dbN) if**:
    - ✅ General-purpose application
    - ✅ Need smooth wavelets
    - ✅ Want N vanishing moments
    - ✅ Orthogonality required
    
    **Choose Coiflets if**:
    - ✅ Signal analysis (not compression)
    - ✅ Need both scaling and wavelet vanishing moments
    - ✅ Better approximation properties needed
    
    **Choose Biorthogonal if**:
    - ✅ Image compression (JPEG 2000)
    - ✅ Need linear phase (symmetric)
    - ✅ Can tolerate redundancy
    
    **Choose Morlet/Mexican Hat if**:
    - ✅ Continuous analysis needed
    - ✅ Time-frequency localization important
    - ✅ Don't need perfect reconstruction
    
    ### Vanishing Moments
    
    **More vanishing moments**:
    - ✅ Better compression of smooth signals
    - ✅ Better approximation
    - ❌ Longer support (more computation)
    - ❌ More coefficients needed
    
    **Rule of thumb**: Use 4-8 vanishing moments for most applications
    """)

with topics[2]:
    st.subheader("Limitations & Trade-offs")
    st.markdown("""
    ### Wavelet Limitations
    
    #### 1. **Heisenberg Uncertainty Principle**
    - Cannot have perfect time AND frequency resolution simultaneously
    - Trade-off: Better time resolution → worse frequency resolution
    - Wavelets provide a good compromise
    
    #### 2. **Boundary Effects**
    - Wavelets near signal boundaries may extend beyond
    - Solutions:
      - Padding (zero, symmetric, periodic)
      - Boundary wavelets
      - Ignore boundary coefficients
    
    #### 3. **Shift Sensitivity**
    - DWT is not shift-invariant
    - Small shifts can change coefficients significantly
    - Solution: Use shift-invariant transforms (undecimated DWT)
    
    #### 4. **Frequency Resolution**
    - Lower frequency resolution than FFT
    - Logarithmic frequency scale (octave bands)
    - May miss closely spaced frequencies
    
    #### 5. **Wavelet Selection**
    - No universal "best" wavelet
    - Must choose based on application
    - Trial and error often needed
    
    ### When NOT to Use Wavelets
    
    ❌ **Stationary signals** - FFT is better
    ❌ **Exact frequency needed** - FFT provides exact frequencies
    ❌ **Very long signals** - Computational cost may be high
    ❌ **Real-time with strict latency** - May be too slow
    """)

with topics[3]:
    st.subheader("Advanced Topics")
    st.markdown("""
    ### 1. Wavelet Packets
    
    **Beyond DWT**: More flexible time-frequency tiling
    
    ```python
    # Wavelet packet decomposition
    wp = pywt.WaveletPacket(data, 'db4', mode='symmetric')
    # Can choose best basis adaptively
    ```
    
    ### 2. Undecimated Wavelet Transform
    
    **Shift-invariant**: Same coefficients regardless of signal shift
    
    ```python
    # Undecimated DWT
    coeffs = pywt.swt(signal, 'db4', level=4)
    ```
    
    ### 3. Lifting Scheme
    
    **Efficient implementation**: Alternative to filter banks
    - Faster computation
    - In-place operations
    - Easier to implement custom wavelets
    
    ### 4. Multiwavelets
    
    **Multiple scaling functions**: Better properties than single wavelets
    - Higher approximation order
    - Better symmetry
    - More complex implementation
    
    ### 5. Complex Wavelets
    
    **Directional information**: Important for 2D signals
    - Dual-tree complex wavelets
    - Better shift invariance
    - Directional selectivity
    
    ### 6. Adaptive Wavelets
    
    **Custom wavelets**: Designed for specific signals
    - Learn from data
    - Optimize for application
    - More complex but potentially better results
    """)

with topics[4]:
    st.subheader("Practical Tips")
    st.markdown("""
    ### 1. Signal Preprocessing
    
    ```python
    # Normalize signal before processing
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    # Handle NaN and Inf
    signal = np.nan_to_num(signal)
    ```
    
    ### 2. Level Selection
    
    ```python
    # Don't exceed maximum level
    max_level = pywt.dwt_max_level(len(signal), wavelet.dec_len)
    level = min(requested_level, max_level)
    ```
    
    ### 3. Mode Selection
    
    ```python
    # Choose appropriate mode for boundaries
    # 'symmetric' - good for smooth signals
    # 'periodic' - good for periodic signals
    # 'zero' - simple but may cause artifacts
    coeffs = pywt.wavedec(signal, 'db4', mode='symmetric')
    ```
    
    ### 4. Threshold Selection
    
    ```python
    # Universal threshold (good starting point)
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # SURE threshold (better for some cases)
    threshold = pywt.threshold(coeffs[-1], mode='sure')
    
    # Minimax threshold (conservative)
    threshold = pywt.threshold(coeffs[-1], mode='minimax')
    ```
    
    ### 5. Memory Management
    
    ```python
    # For large signals, process in chunks
    chunk_size = 10000
    results = []
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]
        coeffs = pywt.wavedec(chunk, 'db4')
        results.append(coeffs)
    ```
    
    ### 6. Visualization Best Practices
    
    ```python
    # Use appropriate colormaps
    # 'jet' - for magnitude (traditional)
    # 'viridis' - perceptually uniform
    # 'hot' - for energy/heat maps
    
    # Adjust aspect ratio for scalograms
    aspect = 'auto'  # Let matplotlib decide
    # or
    aspect = len(time) / len(scales)  # Custom ratio
    ```
    """)

with topics[5]:
    st.subheader("Common Mistakes")
    st.markdown("""
    ### 1. **Using Wrong Wavelet for Reconstruction**
    
    ❌ **Wrong**:
    ```python
    coeffs = pywt.wavedec(signal, 'db4')
    recon = pywt.waverec(coeffs, 'db8')  # Different wavelet!
    ```
    
    ✅ **Correct**:
    ```python
    coeffs = pywt.wavedec(signal, 'db4')
    recon = pywt.waverec(coeffs, 'db4')  # Same wavelet
    ```
    
    ### 2. **Ignoring Length Mismatch**
    
    ❌ **Wrong**:
    ```python
    recon = pywt.waverec(coeffs, 'db4')
    error = signal - recon  # Lengths may differ!
    ```
    
    ✅ **Correct**:
    ```python
    recon = pywt.waverec(coeffs, 'db4')
    if len(recon) != len(signal):
        recon = recon[:len(signal)]
    error = signal - recon
    ```
    
    ### 3. **Thresholding Approximation Coefficients**
    
    ❌ **Wrong**:
    ```python
    coeffs_thresh = [pywt.threshold(c, thresh) for c in coeffs]
    # This thresholds approximation too!
    ```
    
    ✅ **Correct**:
    ```python
    coeffs_thresh = [coeffs[0]]  # Keep approximation
    for c in coeffs[1:]:  # Only threshold details
        coeffs_thresh.append(pywt.threshold(c, thresh))
    ```
    
    ### 4. **Too Many Decomposition Levels**
    
    ❌ **Wrong**:
    ```python
    level = 20  # May exceed max level
    coeffs = pywt.wavedec(signal, 'db4', level=level)
    ```
    
    ✅ **Correct**:
    ```python
    max_level = pywt.dwt_max_level(len(signal), 4)  # db4 has length 4
    level = min(20, max_level)
    coeffs = pywt.wavedec(signal, 'db4', level=level)
    ```
    
    ### 5. **Not Handling Edge Cases**
    
    ❌ **Wrong**:
    ```python
    coeffs = pywt.wavedec(signal, 'db4')  # No error handling
    ```
    
    ✅ **Correct**:
    ```python
    try:
        coeffs = pywt.wavedec(signal, 'db4')
    except ValueError as e:
        print(f"Error: {e}")
        # Handle error appropriately
    ```
    
    ### 6. **Using CWT for Large Signals**
    
    ❌ **Wrong**:
    ```python
    # CWT on very long signal - slow!
    coeffs = pywt.cwt(signal_1million_samples, scales, 'morl')
    ```
    
    ✅ **Correct**:
    ```python
    # Use DWT for processing, CWT for analysis
    coeffs = pywt.wavedec(signal, 'db4')  # Much faster
    # Or downsample first
    signal_down = signal[::10]  # Downsample
    coeffs = pywt.cwt(signal_down, scales, 'morl')
    ```
    """)

st.markdown("""
---

## Key Takeaways

### What Makes Wavelets Powerful

1. **Time-Frequency Localization**: Know both when and what frequency
2. **Multi-Resolution**: Analyze at multiple scales simultaneously
3. **Sparsity**: Efficient representation of many signals
4. **Adaptability**: Different wavelets for different applications

### When to Use Wavelets

✅ Non-stationary signals
✅ Need time-frequency information
✅ Compression is important
✅ Edge detection needed
✅ Multi-scale features present

### Best Practices

1. **Choose appropriate wavelet** for your application
2. **Handle boundaries** properly
3. **Check signal length** vs. wavelet support
4. **Use appropriate levels** (not too many)
5. **Test with your data** - no universal best choice

### Final Thoughts

Wavelets are a powerful tool, but not a panacea. Understanding their strengths and limitations is crucial for effective application. The key is matching the right wavelet and method to your specific problem.

**Remember**: 
- Start simple (Haar, db4)
- Understand your signal
- Test different wavelets
- Consider computational cost
- Validate results

---

*This comprehensive guide should give you a solid foundation for working with wavelets in your own projects!*
""")



