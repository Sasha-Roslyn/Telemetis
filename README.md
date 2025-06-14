OdysseyNet is a convolutional autoencoder designed for geoscientific image reconstruction and feature extraction. Originally developed to process 2D mineralogical images such as MLA or BSE outputs, it supports a variety of preprocessing modes, including whole-image, masked, patched, and isolated particle inputs.
Developed and trained to reconstruct whole images and predict mineral groups of interest based on pre-processed images (where the major mineral group is preserved and the mineral groups of interest are blank). The network has a very deep and complex structure due to the complex nature of the input images it was trained on. It is recommended to adjust the depth of the network according to your input data while preserving the architecture.
Two versions of OdysseyNet are shared:

odysseynet_dual: Predicts mineral groups using dual-input approach
odysseynet_abundant_sparse: Works with inputs of different density characteristics

Note: These are simplified reference implementations for research and educational purposes. Production systems may require additional considerations for industrial applications.

OdysseyNet forms part of the Telemetis platform — an AI-powered geoscientific system.

Features:

- Trainable convolutional autoencoder in TensorFlow
- Compatible with multi-channel TIFF and MLA image data
- Streamlit interface modules for loading, training, evaluation, and visual comparison (available as part of the Peneloop module)
- Extracts feature vectors from the bottleneck layer for downstream clustering (available in Peneloop and Olympos modules)
- Custom loss functions combining MSE, SSIM, and Edge preservation
- Support for both abundant (3-channel) and sparse (2-channel) mineral data

Quick Start:

Clone the repository:

git clone https://github.com/Sasha-Roslyn/Telemetis.git
cd Telemetis

Install dependencies:

pip install -r requirements.txt

Requirements:

tensorflow>=2.8.0
numpy>=1.19.0
opencv-python>=4.5.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
tifffile>=2021.7.2
pillow>=8.0.0

Basic Usage: 
Dual-Input Approach:

```python
from odysseynet_dual import build_odysseynet_dual, combined_loss

# Create dual-input model (binary + RGB)
model = build_odysseynet_dual(
    binary_shape=(128, 128, 5), 
    rgb_shape=(128, 128, 3)
)
```
# Model comes pre-compiled with custom loss functions
# Ready for training with your data

Abundant/Sparse Approach:

python
from odysseynet_abundant_sparse import build_abundant_autoencoder, build_sparse_autoencoder

# For abundant (dense) 3-channel data
abundant_model = build_abundant_autoencoder(input_shape=(128, 128, 3))

# For sparse (low-density) 2-channel data  
sparse_model = build_sparse_autoencoder(input_shape=(128, 128, 2))

Feature Extraction:
python
# Extract features from trained model
from odysseynet_dual import extract_feature_vectors_dual

features = extract_feature_vectors_dual(model, binary_data, rgb_data)
print(f"Feature vector shape: {features.shape}")

Repository Structure:

Telemetis/
├── odysseynet_dual.py                    # Dual-input autoencoder implementation
├── odysseynet_abundant_sparse.py         # Abundant/sparse autoencoder variants
├── requirements.txt                      # Dependencies
├── LICENSE                              # CC BY-NC 4.0 license
└── README.md                            # This file

Model Architectures:

Dual-Input Model

- Binary Input: 5-channel mineral classification data
- RGB Input: 3-channel preprocessed images with gaps
- Output: Complete RGB reconstruction
- Loss: Combined MSE + SSIM + Edge preservation

Abundant/Sparse Models:

- Abundant: Optimized for dense 3-channel data with LeakyReLU activation
- Sparse: Optimized for sparse 2-channel data with larger filters
- Adaptive: Different loss weightings for different data densities

Integration:

OdysseyNet integrates with:

- Peneloop: Preprocessing, masking, and binarization pipelines
- Olympos: SOM + PCA clustering of extracted features
- Athena: Downstream regression and simulation workflows

Performance:

Typical performance metrics on geological datasets:

- PSNR: 28-35 dB (depending on approach and data complexity)
- SSIM: 0.85-0.95 (structural similarity preservation)
- Training time: 30-60 minutes (depending on dataset size and hardware)

License:

Released under the CC BY-NC 4.0 License.

Citation:

If you use OdysseyNet in your research, please consider citing this repository:

bibtex@software{odysseynet2025,
  author = {Roslin, Alexandra},
  title = {OdysseyNet: Convolutional Autoencoders for Geoscientific Image Reconstruction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Sasha-Roslyn/Telemetis}
}
Credits:

Created by: Sasha Roslin
Acknowledgments: BHP for input data and HPC facility access to train OdysseyNet.

The name OdysseyNet reflects a journey through complex image space toward compact, expressive understanding.
