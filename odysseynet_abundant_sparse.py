
#!/usr/bin/env python3
"""
OdysseyNet Abundant & Sparse - Multi-Density Channel Processing
Specialised autoencoders for different data densities: abundant (3-channel) vs sparse (2-channel)

Key Features:
- Abundant model: Optimised for dense 3-channel data
- Sparse model: Optimised for sparse 2-channel data  
- Different architectures for different data characteristics
- Custom loss functions with density-specific weights
- Comparative training and evaluation

License: CC BY-NC 4.0
Author: Sasha Roslin
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Custom loss functions and metrics
def ssim_loss(y_true, y_pred):
    """Structural Similarity Index loss"""
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def psnr_metric(y_true, y_pred):
    """Peak Signal-to-Noise Ratio metric"""
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ms_ssim_loss(y_true, y_pred):
    """Multi-Scale Structural Similarity Index loss"""
    return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0))

def combined_loss_abundant(y_true, y_pred):
    """Combined loss for abundant data (higher SSIM weight)"""
    alpha = 0.6  # Higher weight for MS-SSIM
    mse = MeanSquaredError()
    return alpha * ms_ssim_loss(y_true, y_pred) + (1-alpha) * mse(y_true, y_pred)

def combined_loss_sparse(y_true, y_pred):
    """Combined loss for sparse data (lower SSIM weight)"""
    alpha = 0.4  # Lower weight for MS-SSIM
    mse = MeanSquaredError()
    return alpha * ms_ssim_loss(y_true, y_pred) + (1-alpha) * mse(y_true, y_pred)

def build_abundant_autoencoder(input_shape=(128, 128, 3)):
    """
    Builds autoencoder optimised for abundant (dense) 3-channel data
    
    Uses LeakyReLU in encoder for better gradient flow with dense features
    Smaller initial filter sizes due to rich feature content
    
    Args:
        input_shape: Shape of input images (height, width, 3)
    
    Returns:
        Compiled Keras model for abundant data
    """
    
    input_abundant = Input(shape=input_shape, name="Input_Abundant")
    dropout_rate = 0.3
    dropout_deep = 0.5
    
    # Abundant Encoder - smaller filters, LeakyReLU for dense features
    encoder = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_uniform')(input_abundant)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU(alpha=0.1)(encoder)
    
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU(alpha=0.1)(encoder)
    encoder = Dropout(dropout_rate)(encoder)
    
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU(alpha=0.1)(encoder)
    encoder = Dropout(dropout_rate)(encoder)
    
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU(alpha=0.1)(encoder)
    encoder = Dropout(dropout_deep)(encoder)
    
    # Bottleneck
    bottleneck = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.1), padding='same', 
                       kernel_initializer='he_uniform')(encoder)
    bottleneck = BatchNormalization()(bottleneck)
    
    # Decoder - ReLU for stable reconstruction
    decoder = Conv2DTranspose(128, (3, 3), strides=2, padding='same', 
                             kernel_initializer='he_uniform')(bottleneck)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    
    decoder = Conv2DTranspose(64, (3, 3), strides=2, padding='same', 
                             kernel_initializer='he_uniform')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    
    decoder = Conv2DTranspose(32, (3, 3), strides=2, padding='same', 
                             kernel_initializer='he_uniform')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    
    output = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name="Output_Abundant")(decoder)
    
    model = Model(inputs=input_abundant, outputs=output, name="Abundant_Autoencoder")
    
    # Compile with abundant-specific settings
    model.compile(
        optimizer=Adam(learning_rate=0.00001, clipvalue=0.5),
        loss=combined_loss_abundant,
        metrics=[psnr_metric, ssim_loss, ms_ssim_loss, 'mae']
    )
    
    return model

def build_sparse_autoencoder(input_shape=(128, 128, 2)):
    """
    Builds autoencoder optimised for sparse (low-density) 2-channel data
    
    Uses larger initial filter sizes to capture sparse features
    Regular ReLU throughout for stable sparse feature learning
    
    Args:
        input_shape: Shape of input images (height, width, 2)
    
    Returns:
        Compiled Keras model for sparse data
    """
    
    input_sparse = Input(shape=input_shape, name="Input_Sparse")
    dropout_rate = 0.3
    dropout_deep = 0.5
    
    # Sparse Encoder - larger filters, ReLU for sparse features
    encoder = Conv2D(32, (3, 3), padding='same')(input_sparse)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = Conv2D(64, (3, 3), padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Dropout(dropout_rate)(encoder)
    
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = Conv2D(128, (3, 3), padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Dropout(dropout_deep)(encoder)
    
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = Conv2D(256, (3, 3), padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Dropout(dropout_deep)(encoder)
    
    # Bottleneck - larger for sparse feature representation
    bottleneck = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder)
    bottleneck = BatchNormalization()(bottleneck)
    
    # Decoder
    decoder = Conv2DTranspose(256, (3, 3), strides=2, padding='same')(bottleneck)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    
    decoder = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    
    decoder = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    
    output = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name="Output_Sparse")(decoder)
    
    model = Model(inputs=input_sparse, outputs=output, name="Sparse_Autoencoder")
    
    # Compile with sparse-specific settings
    model.compile(
        optimizer=Adam(learning_rate=0.00001, clipvalue=0.5),
        loss=combined_loss_sparse,
        metrics=[psnr_metric, ssim_loss, ms_ssim_loss, 'mae']
    )
    
    return model

def generate_abundant_data(num_samples=600, input_shape=(128, 128, 3)):
    """
    Generate synthetic abundant (dense) data
    Rich patterns with high inter-channel correlation
    """
    
    data = []
    
    for i in range(num_samples):
        image = np.zeros(input_shape)
        
        # Create dense, correlated patterns
        x, y = np.meshgrid(np.linspace(0, 6*np.pi, input_shape[0]), 
                         np.linspace(0, 6*np.pi, input_shape[1]))
        
        # Channel 1: Primary dense pattern
        pattern1 = 0.6 + 0.3 * np.sin(x/2) * np.cos(y/2) + 0.1 * np.sin(x) * np.cos(y)
        
        # Channel 2: Correlated secondary pattern  
        pattern2 = 0.5 + 0.2 * np.sin(x/3 + np.pi/4) * np.cos(y/3) + 0.15 * pattern1
        
        # Channel 3: Complex correlated pattern
        pattern3 = 0.4 + 0.2 * np.cos(x/4) * np.sin(y/4) + 0.1 * pattern1 + 0.1 * pattern2
        
        image[:, :, 0] = np.clip(pattern1, 0, 1)
        image[:, :, 1] = np.clip(pattern2, 0, 1)
        image[:, :, 2] = np.clip(pattern3, 0, 1)
        
        # Add correlated noise
        noise = 0.05 * np.random.normal(0, 1, input_shape)
        image += noise
        image = np.clip(image, 0, 1)
        
        data.append(image)
    
    return np.array(data)

def generate_sparse_data(num_samples=600, input_shape=(128, 128, 2)):
    """
    Generate synthetic sparse (low-density) data
    Sparse patterns with localised features
    """
    
    data = []
    
    for i in range(num_samples):
        image = np.zeros(input_shape)
        
        # Create sparse, localised patterns
        x, y = np.meshgrid(np.linspace(0, 4*np.pi, input_shape[0]), 
                         np.linspace(0, 4*np.pi, input_shape[1]))
        
        # Channel 1: Sparse primary features
        pattern1 = 0.3 + 0.2 * np.sin(x) * np.cos(y)
        pattern1 = np.where(pattern1 > 0.4, pattern1, 0.1)  # Sparsify
        
        # Channel 2: Sparse secondary features
        center_x, center_y = input_shape[0]//2, input_shape[1]//2
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        pattern2 = 0.2 + 0.15 * np.sin(radius/8)
        pattern2 = np.where(pattern2 > 0.25, pattern2, 0.05)  # Sparsify
        
        image[:, :, 0] = np.clip(pattern1, 0, 1)
        image[:, :, 1] = np.clip(pattern2, 0, 1)
        
        # Add sparse noise
        noise = 0.03 * np.random.normal(0, 1, input_shape)
        image += noise
        image = np.clip(image, 0, 1)
        
        data.append(image)
    
    return np.array(data)

def train_model_with_callbacks(model, X_train, X_val, model_name, epochs=50, batch_size=16):
    """
    Train model with proper callbacks
    """
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, 
                                  verbose=1, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(f"{model_name}_checkpoint.h5", save_weights_only=True, 
                                        monitor="val_loss", save_best_only=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    
    # Train
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint_callback, lr_schedule],
        verbose=1
    )
    
    return history

def plot_comparison_results(abundant_model, sparse_model, abundant_test, sparse_test, num_samples=3):
    """
    Compare reconstruction results between abundant and sparse models
    """
    
    abundant_pred = abundant_model.predict(abundant_test[:num_samples])
    sparse_pred = sparse_model.predict(sparse_test[:num_samples])
    
    fig, axes = plt.subplots(4, num_samples, figsize=(15, 12))
    
    for i in range(num_samples):
        # Abundant original
        axes[0, i].imshow(abundant_test[i])
        axes[0, i].set_title(f'Abundant Original {i+1}')
        axes[0, i].axis('off')
        
        # Abundant reconstructed
        axes[1, i].imshow(abundant_pred[i])
        axes[1, i].set_title(f'Abundant Reconstructed {i+1}')
        axes[1, i].axis('off')
        
        # Sparse original
        axes[2, i].imshow(sparse_test[i], cmap='gray')
        axes[2, i].set_title(f'Sparse Original {i+1}')
        axes[2, i].axis('off')
        
        # Sparse reconstructed
        axes[3, i].imshow(sparse_pred[i], cmap='gray')
        axes[3, i].set_title(f'Sparse Reconstructed {i+1}')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_comparison(abundant_history, sparse_history):
    """
    Compare training metrics between abundant and sparse models
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss comparison
    axes[0, 0].plot(abundant_history.history['loss'], label='Abundant Training', color='blue')
    axes[0, 0].plot(abundant_history.history['val_loss'], label='Abundant Validation', color='blue', linestyle='--')
    axes[0, 0].plot(sparse_history.history['loss'], label='Sparse Training', color='red')
    axes[0, 0].plot(sparse_history.history['val_loss'], label='Sparse Validation', color='red', linestyle='--')
    axes[0, 0].set_title('Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE comparison
    axes[0, 1].plot(abundant_history.history['mae'], label='Abundant Training', color='blue')
    axes[0, 1].plot(abundant_history.history['val_mae'], label='Abundant Validation', color='blue', linestyle='--')
    axes[0, 1].plot(sparse_history.history['mae'], label='Sparse Training', color='red')
    axes[0, 1].plot(sparse_history.history['val_mae'], label='Sparse Validation', color='red', linestyle='--')
    axes[0, 1].set_title('MAE Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PSNR comparison
    axes[1, 0].plot(abundant_history.history['psnr_metric'], label='Abundant Training', color='blue')
    axes[1, 0].plot(abundant_history.history['val_psnr_metric'], label='Abundant Validation', color='blue', linestyle='--')
    axes[1, 0].plot(sparse_history.history['psnr_metric'], label='Sparse Training', color='red')
    axes[1, 0].plot(sparse_history.history['val_psnr_metric'], label='Sparse Validation', color='red', linestyle='--')
    axes[1, 0].set_title('PSNR Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # SSIM comparison  
    axes[1, 1].plot(abundant_history.history['ssim_loss'], label='Abundant Training', color='blue')
    axes[1, 1].plot(abundant_history.history['val_ssim_loss'], label='Abundant Validation', color='blue', linestyle='--')
    axes[1, 1].plot(sparse_history.history['ssim_loss'], label='Sparse Training', color='red')
    axes[1, 1].plot(sparse_history.history['val_ssim_loss'], label='Sparse Validation', color='red', linestyle='--')
    axes[1, 1].set_title('SSIM Loss Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('SSIM Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def extract_features_abundant(model, X):
    """Extract features from abundant model bottleneck"""
    # Find bottleneck layer (256 filters)
    for layer in model.layers:
        if 'conv2d' in layer.name and hasattr(layer, 'filters') and layer.filters == 256:
            encoder = Model(model.input, layer.output)
            return encoder.predict(X)
    return None

def extract_features_sparse(model, X):
    """Extract features from sparse model bottleneck"""
    # Find bottleneck layer (512 filters)
    for layer in model.layers:
        if 'conv2d' in layer.name and hasattr(layer, 'filters') and layer.filters == 512:
            encoder = Model(model.input, layer.output)
            return encoder.predict(X)
    return None

def example_usage():
    """
    Demonstrate abundant vs sparse autoencoder training and comparison
    """
    
    print("OdysseyNet Abundant & Sparse - Multi-Density Processing Demo")
    print("=" * 60)
    
    # Configuration
    ABUNDANT_SHAPE = (128, 128, 3)
    SPARSE_SHAPE = (128, 128, 2)
    NUM_SAMPLES = 500
    EPOCHS = 40
    BATCH_SIZE = 16
    
    print(f"Configuration:")
    print(f"- Abundant Shape: {ABUNDANT_SHAPE}")
    print(f"- Sparse Shape: {SPARSE_SHAPE}")
    print(f"- Samples per type: {NUM_SAMPLES}")
    print(f"- Training Epochs: {EPOCHS}")
    print(f"- Batch Size: {BATCH_SIZE}")
    print()
    
    # Generate data
    print("Generating synthetic data...")
    print("- Creating abundant (dense) data...")
    abundant_data = generate_abundant_data(NUM_SAMPLES, ABUNDANT_SHAPE)
    
    print("- Creating sparse (low-density) data...")
    sparse_data = generate_sparse_data(NUM_SAMPLES, SPARSE_SHAPE)
    
    # Split data
    abundant_train, abundant_test = train_test_split(abundant_data, test_size=0.2, random_state=42)
    abundant_train, abundant_val = train_test_split(abundant_train, test_size=0.2, random_state=42)
    
    sparse_train, sparse_test = train_test_split(sparse_data, test_size=0.2, random_state=42)
    sparse_train, sparse_val = train_test_split(sparse_train, test_size=0.2, random_state=42)
    
    print(f"Data split complete:")
    print(f"- Abundant: {len(abundant_train)} train, {len(abundant_val)} val, {len(abundant_test)} test")
    print(f"- Sparse: {len(sparse_train)} train, {len(sparse_val)} val, {len(sparse_test)} test")
    print()
    
    # Build models
    print("Building models...")
    abundant_model = build_abundant_autoencoder(ABUNDANT_SHAPE)
    sparse_model = build_sparse_autoencoder(SPARSE_SHAPE)
    
    print(f"Abundant model parameters: {abundant_model.count_params():,}")
    print(f"Sparse model parameters: {sparse_model.count_params():,}")
    print()
    
    # Train models
    print("Training abundant model...")
    abundant_history = train_model_with_callbacks(
        abundant_model, abundant_train, abundant_val, "abundant_model", EPOCHS, BATCH_SIZE
    )
    
    print("\nTraining sparse model...")
    sparse_history = train_model_with_callbacks(
        sparse_model, sparse_train, sparse_val, "sparse_model", EPOCHS, BATCH_SIZE
    )
    
    # Evaluate models
    print("\nEvaluating models...")
    abundant_loss = abundant_model.evaluate(abundant_test, abundant_test, verbose=0)
    sparse_loss = sparse_model.evaluate(sparse_test, sparse_test, verbose=0)
    
    print(f"Abundant Test Loss: {abundant_loss:.4f}")
    print(f"Sparse Test Loss: {sparse_loss:.4f}")
    
    # Extract features
    print("\nExtracting features...")
    abundant_features = extract_features_abundant(abundant_model, abundant_test[:5])
    sparse_features = extract_features_sparse(sparse_model, sparse_test[:5])
    
    if abundant_features is not None:
        print(f"Abundant feature shape: {abundant_features.shape}")
    if sparse_features is not None:
        print(f"Sparse feature shape: {sparse_features.shape}")
    
    # Visualize results
    print("\nVisualizing results...")
    plot_training_comparison(abundant_history, sparse_history)
    plot_comparison_results(abundant_model, sparse_model, abundant_test, sparse_test)
    
    # Save models
    abundant_model.save("odysseynet_abundant_model.keras")
    sparse_model.save("odysseynet_sparse_model.keras")
    print("\nModels saved:")
    print("- odysseynet_abundant_model.keras")
    print("- odysseynet_sparse_model.keras")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("Key concepts demonstrated:")
    print("• Abundant vs Sparse data processing")
    print("• Density-specific architectures")
    print("• Custom loss function weighting")
    print("• Comparative model evaluation")
    print("• Feature extraction from different densities")
    print("=" * 60)

if __name__ == "__main__":
    example_usage()