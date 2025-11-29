import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mathematical Foundation", page_icon="🔢", layout="wide")

st.title("🔢 Mathematical Foundation of Wavelets")

st.markdown("""
## Core Mathematical Concepts

### 1. Wavelet Function (Mother Wavelet)

A **mother wavelet** ψ(t) is a function that satisfies:

#### Admissibility Condition
""")

st.latex(r"""
\int_{-\infty}^{\infty} \psi(t) \, dt = 0
""")

st.markdown("""
This means the wavelet has zero mean (equal positive and negative areas).

#### Energy Normalization
""")

st.latex(r"""
\int_{-\infty}^{\infty} |\psi(t)|^2 \, dt = 1
""")

st.markdown("""
The wavelet has unit energy.

#### Decay Condition
""")

st.latex(r"""
\int_{-\infty}^{\infty} \frac{|\hat{\psi}(\omega)|^2}{|\omega|} \, d\omega < \infty
""")

st.markdown("""
where $\hat{\psi}(\omega)$ is the Fourier transform of ψ(t). This ensures the wavelet transform is invertible.

### 2. Wavelet Family

From a mother wavelet, we create a family of wavelets through **scaling** and **translation**:
""")

st.latex(r"""
\psi_{a,b}(t) = \frac{1}{\sqrt{a}} \psi\left(\frac{t-b}{a}\right)
""")

st.markdown("""
where:
- **a > 0** is the **scale parameter** (dilation)
  - Large a → stretched wavelet (low frequency)
  - Small a → compressed wavelet (high frequency)
- **b** is the **translation parameter** (shift)
  - Controls the position of the wavelet in time

### 3. Continuous Wavelet Transform (CWT)

The CWT of a signal f(t) is defined as:
""")

st.latex(r"""
W_f(a, b) = \int_{-\infty}^{\infty} f(t) \overline{\psi_{a,b}(t)} \, dt = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} f(t) \overline{\psi\left(\frac{t-b}{a}\right)} \, dt
""")

st.markdown("""
where $\overline{\psi}$ denotes the complex conjugate.

**Interpretation**: W_f(a, b) measures the similarity between the signal and the wavelet at scale a and position b.

### 4. Inverse Wavelet Transform

The original signal can be reconstructed from its wavelet transform:
""")

st.latex(r"""
f(t) = \frac{1}{C_\psi} \int_0^{\infty} \int_{-\infty}^{\infty} W_f(a, b) \psi_{a,b}(t) \frac{da \, db}{a^2}
""")

st.markdown("""
where $C_\psi$ is the admissibility constant.

### 5. Discrete Wavelet Transform (DWT)

For practical applications, we use discrete scales and translations:

#### Dyadic Grid
""")

st.latex(r"""
a = 2^j, \quad b = k \cdot 2^j
""")

st.markdown("""
where j, k ∈ ℤ (integers).

The discrete wavelets become:
""")

st.latex(r"""
\psi_{j,k}(t) = 2^{-j/2} \psi(2^{-j}t - k)
""")

st.markdown("""
### 6. Multi-Resolution Analysis (MRA)

MRA is the theoretical foundation of discrete wavelets. It decomposes a signal into nested subspaces:

""")

st.latex(r"""
\cdots \subset V_{-1} \subset V_0 \subset V_1 \subset V_2 \subset \cdots
""")

st.markdown("""
where:
- **V_j** = space of functions at resolution level j
- **W_j** = detail space (orthogonal complement of V_j in V_{j+1})

#### Decomposition
""")

st.latex(r"""
V_{j+1} = V_j \oplus W_j
""")

st.markdown("""
This means any function in V_{j+1} can be written as:
- A **coarse approximation** in V_j
- Plus **details** in W_j

### 7. Scaling Function (Father Wavelet)

The scaling function φ(t) generates the approximation spaces V_j:
""")

st.latex(r"""
V_j = \text{span}\{\phi_{j,k}(t) = 2^{-j/2}\phi(2^{-j}t - k) : k \in \mathbb{Z}\}
""")

st.markdown("""
#### Two-Scale Relation
""")

st.latex(r"""
\phi(t) = \sqrt{2} \sum_k h_k \phi(2t - k)
""")

st.latex(r"""
\psi(t) = \sqrt{2} \sum_k g_k \phi(2t - k)
""")

st.markdown("""
where:
- **h_k** = low-pass filter coefficients (scaling function)
- **g_k** = high-pass filter coefficients (wavelet function)

### 8. Fast Wavelet Transform (FWT)

The DWT can be computed efficiently using filter banks:

#### Forward Transform (Analysis)
""")

st.latex(r"""
c_{j,k} = \sum_n h_{n-2k} c_{j+1,n} \quad \text{(approximation)}
""")

st.latex(r"""
d_{j,k} = \sum_n g_{n-2k} c_{j+1,n} \quad \text{(details)}
""")

st.markdown("""
#### Inverse Transform (Synthesis)
""")

st.latex(r"""
c_{j+1,n} = \sum_k h_{n-2k} c_{j,k} + \sum_k g_{n-2k} d_{j,k}
""")

st.markdown("""
**Complexity**: O(N) for a signal of length N (much faster than FFT's O(N log N) for certain operations).

### 9. Wavelet Properties

#### Orthogonality
""")

st.latex(r"""
\int_{-\infty}^{\infty} \psi_{j,k}(t) \overline{\psi_{j',k'}(t)} \, dt = \delta_{j,j'} \delta_{k,k'}
""")

st.markdown("""
Orthogonal wavelets provide non-redundant representations.

#### Vanishing Moments

A wavelet has **p vanishing moments** if:
""")

st.latex(r"""
\int_{-\infty}^{\infty} t^m \psi(t) \, dt = 0, \quad m = 0, 1, \ldots, p-1
""")

st.markdown("""
**Importance**: More vanishing moments → better compression of smooth signals.

### 10. Common Wavelet Families

#### Haar Wavelet
- Simplest wavelet
- 1 vanishing moment
- Discontinuous (not smooth)

#### Daubechies Wavelets (dbN)
- N vanishing moments
- Compact support
- Orthogonal
- Smooth for N ≥ 4

#### Coiflets
- Both scaling function and wavelet have vanishing moments
- Better symmetry than Daubechies

#### Biorthogonal Wavelets
- Linear phase (symmetric)
- Used in image compression (JPEG 2000)

### 11. Wavelet Packet Transform

Generalization of DWT that provides more flexible time-frequency tiling:

- DWT: Fixed decomposition tree
- Wavelet Packets: Adaptive decomposition tree

### 12. 2D Wavelet Transform

For images, we apply 1D wavelets along rows and columns:

""")

st.latex(r"""
W_{LL}, W_{LH}, W_{HL}, W_{HH}
""")

st.markdown("""
where:
- **LL**: Low-Low (approximation)
- **LH**: Low-High (horizontal details)
- **HL**: High-Low (vertical details)
- **HH**: High-High (diagonal details)

### Mathematical Example: Haar Wavelet

Let's verify the Haar wavelet satisfies the admissibility condition:
""")

# Visual demonstration
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

t = np.linspace(-0.5, 1.5, 1000)
haar = np.zeros_like(t)
haar[(t >= 0) & (t < 0.5)] = 1
haar[(t >= 0.5) & (t < 1)] = -1

# Show that integral is zero
integral = np.trapz(haar, t)
axes[0].plot(t, haar, 'b-', linewidth=2, label=f'Haar Wavelet')
axes[0].fill_between(t, 0, haar, where=(haar >= 0), alpha=0.3, color='green', label='Positive area')
axes[0].fill_between(t, 0, haar, where=(haar < 0), alpha=0.3, color='red', label='Negative area')
axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[0].set_title(f'Haar Wavelet (Integral ≈ {integral:.2e})', fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Amplitude')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Show scaling and translation
t2 = np.linspace(-1, 3, 2000)
haar_original = np.zeros_like(t2)
haar_original[(t2 >= 0) & (t2 < 0.5)] = 1
haar_original[(t2 >= 0.5) & (t2 < 1)] = -1

haar_scaled = np.zeros_like(t2)
haar_scaled[(t2 >= 0) & (t2 < 1)] = 0.5  # a=2, so 1/sqrt(2) amplitude
haar_scaled[(t2 >= 1) & (t2 < 2)] = -0.5

haar_translated = np.zeros_like(t2)
haar_translated[(t2 >= 1) & (t2 < 1.5)] = 1
haar_translated[(t2 >= 1.5) & (t2 < 2)] = -1

axes[1].plot(t2, haar_original, 'b-', linewidth=2, label='Original (a=1, b=0)')
axes[1].plot(t2, haar_scaled, 'r--', linewidth=2, label='Scaled (a=2, b=0)')
axes[1].plot(t2, haar_translated, 'g:', linewidth=2, label='Translated (a=1, b=1)')
axes[1].set_title('Scaling and Translation', fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Amplitude')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
### Code: Verifying Wavelet Properties

```python
import numpy as np
from scipy import integrate

# Define Haar wavelet
def haar_wavelet(t):
    result = np.zeros_like(t)
    result[(t >= 0) & (t < 0.5)] = 1
    result[(t >= 0.5) & (t < 1)] = -1
    return result

# Verify admissibility (zero mean)
t = np.linspace(-1, 2, 10000)
psi = haar_wavelet(t)
integral = np.trapz(psi, t)
print(f"Integral (should be ~0): {integral}")

# Verify energy normalization
energy = np.trapz(psi**2, t)
print(f"Energy (should be 1): {energy}")

# Verify vanishing moment (m=0)
moment_0 = np.trapz(psi, t)
print(f"0th moment (should be 0): {moment_0}")
```

### Summary

The mathematical foundation of wavelets rests on:
1. **Admissibility conditions** - Ensures invertibility
2. **Scaling and translation** - Creates wavelet families
3. **Multi-resolution analysis** - Theoretical framework
4. **Filter banks** - Efficient computation
5. **Orthogonality** - Non-redundant representation

These mathematical properties enable wavelets to provide efficient, localized time-frequency analysis that Fourier transforms cannot match.
""")


