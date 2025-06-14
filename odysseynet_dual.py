
#!/usr/bin/env python3
"""
OdysseyNet Dual - Dual-Input Autoencoder for Multi-Channel Mineral Reconstruction
An implementation of the dual-input architecture for learning from binary and RGB inputs

Key Features:
- Dual-input architecture (binary + preprocessed RGB)
- Custom loss functions (MSE + SSIM + Edge)
- Feature learning from concatenated inputs
- Gap filling and reconstruction

License: CC BY-NC 4.0
Author: Sasha Roslin
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Activation, concatenate
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def edge_loss(y_true, y_pred):
    """Edge preservation loss using Sobel edge detection"""
    sobel_true = tf.image.sobel_edges(y_true)
    sobel_pred = tf.image.sobel_edges(y_pred)
    return MeanSquaredError()(sobel_true, sobel_pred)

def combined_loss(y_true, y_pred, alpha=1.0, beta=0.0, max_val=1.0):
    """
    Combined loss function: MSE + SSIM + Edge Loss
    
    Args:
        alpha: Weight for MSE loss
        beta: Weight for SSIM loss
        Remaining weight (1-alpha-beta) goes to edge loss
    """
    mse_loss = MeanSquaredError()(y_true, y_pred)
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_val))
    edge_loss_val = edge_loss(y_true, y_pred)
    
    return alpha * mse_loss + beta * ssim_loss + (1 - alpha - beta) * edge_loss_val

def build_odysseynet_dual(binary_shape=(128, 128, 5), rgb_shape=(128, 128, 3)):
    """
    Builds a dual-input autoencoder (OdysseyNet Dual)
    
    Args:
        binary_shape: Shape of binary input (height, width, binary_channels)
        rgb_shape: Shape of RGB input (height, width, 3)
    
    Returns:
        Compiled Keras model
    """
    
    # Dual inputs
    input_binary = Input(shape=binary_shape, name="Input_Binary")
    input_preprocessed = Input(shape=rgb_shape, name="Input_Preprocessed")
    
    # Merge inputs
    merged_input = concatenate([input_binary, input_preprocessed])
    
    # Encoder
    encoder = Conv2D(32, (3, 3), padding='same')(merged_input)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    
    encoder = Conv2D(64, (3, 3), padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    
    encoder = Conv2D(128, (3, 3), padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    
    encoder = Conv2D(256, (3, 3), padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    
    # Bottleneck
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
    
    decoder = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    
    # Output
    output = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name="Output")(decoder)
    
    # Construct the autoencoder model
    autoencoder = Model(inputs=[input_binary, input_preprocessed], outputs=output, 
                       name="OdysseyNet_Dual_Autoencoder")
    
    # Compile the model
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=combined_loss,
        metrics=[edge_loss, 'mae']
    )
    
    return autoencoder

def generate_dual_input_data(num_samples=800, binary_shape=(128, 128, 5), rgb_shape=(128, 128, 3)):
    """
    Generate synthetic dual-input data for demonstration
    
    Args:
        num_samples: Number of samples to generate
        binary_shape: Shape of binary input
        rgb_shape: Shape of RGB input
    
    Returns:
        binary_data: Binary multi-channel input
        preprocessed_data: RGB input with gaps
        target_data: Complete RGB target
    """
    
    binary_data = []
    preprocessed_data = []
    target_data = []
    
    print(f"Generating {num_samples} dual-input samples...")
    
    for i in range(num_samples):
        # Create binary channels
        binary_img = np.zeros(binary_shape)
        
        for c in range(binary_shape[2]):
            x, y = np.meshgrid(np.linspace(0, 4*np.pi, binary_shape[0]), 
                             np.linspace(0, 4*np.pi, binary_shape[1]))
            
            if c == 0:  # Primary channel
                pattern = 0.6 + 0.3 * np.sin(x/2) * np.cos(y/2)
            elif c == 1:  # Secondary channel
                pattern = 0.5 + 0.2 * np.sin(x/3) * np.cos(y/3)
            elif c == 2:  # Tertiary channel
                center_x, center_y = binary_shape[0]//2, binary_shape[1]//2
                radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                pattern = 0.4 + 0.2 * np.sin(radius/8)
            else:  # Additional channels
                pattern = 0.3 + 0.1 * np.random.normal(0, 1, binary_shape[:2])
            
            binary_img[:, :, c] = (pattern > 0.5).astype(np.float32)
        
        # Create target RGB from binary channels
        target_rgb = np.zeros(rgb_shape)
        target_rgb[:, :, 0] = 0.6 * binary_img[:, :, 0] + 0.4 * binary_img[:, :, 1]  # Red
        target_rgb[:, :, 1] = 0.5 * binary_img[:, :, 1] + 0.3 * binary_img[:, :, 2] + 0.2 * binary_img[:, :, 3]  # Green
        target_rgb[:, :, 2] = 0.3 * binary_img[:, :, 0] + 0.2 * binary_img[:, :, 2] + 0.5 * binary_img[:, :, 4]  # Blue
        
        # Add noise and normalize
        target_rgb += 0.05 * np.random.normal(0, 1, target_rgb.shape)
        target_rgb = np.clip(target_rgb, 0, 1)
        
        # Create preprocessed RGB with gaps (white regions for missing data)
        primary_channel = np.argmax(np.sum(binary_img, axis=(0, 1)))
        preprocessed_rgb = np.ones(rgb_shape)  # Start with white
        
        # Keep only primary mineral information, rest as white gaps
        primary_mask = binary_img[:, :, primary_channel]
        for c in range(3):
            preprocessed_rgb[:, :, c] = np.where(
                primary_mask > 0.5,
                target_rgb[:, :, c],
                1.0  # White gaps for missing information
            )
        
        # Add slight noise
        preprocessed_rgb += 0.02 * np.random.normal(0, 1, preprocessed_rgb.shape)
        preprocessed_rgb = np.clip(preprocessed_rgb, 0, 1)
        
        binary_data.append(binary_img)
        preprocessed_data.append(preprocessed_rgb)
        target_data.append(target_rgb)
        
        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")
    
    return np.array(binary_data), np.array(preprocessed_data), np.array(target_data)

def train_odysseynet_dual(model, binary_train, rgb_train, target_train, 
                         binary_val, rgb_val, target_val, 
                         epochs=100, batch_size=16):
    """
    Trains the OdysseyNet Dual model
    
    Args:
        model: OdysseyNet Dual model to train
        binary_train, rgb_train, target_train: Training data
        binary_val, rgb_val, target_val: Validation data
        epochs: Number of training epochs
        batch_size: Training batch size
    
    Returns:
        Training history
    """
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint("odysseynet_dual_checkpoint.h5", save_weights_only=True, 
                                         monitor="val_loss", save_best_only=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    
    # Training
    history = model.fit(
        x=[binary_train, rgb_train],
        y=target_train,
        validation_data=([binary_val, rgb_val], target_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint_callback, lr_schedule],
        verbose=1
    )
    
    return history

def extract_feature_vectors_dual(model, binary_data, rgb_data):
    """
    Extracts feature vectors from the encoder bottleneck
    
    Args:
        model: Trained OdysseyNet Dual model
        binary_data: Binary input data
        rgb_data: RGB input data
    
    Returns:
        Feature vectors from encoder bottleneck
    """
    # Find the bottleneck layer (layer with name containing 'bottleneck' or the deepest conv layer)
    bottleneck_layer = None
    for i, layer in enumerate(model.layers):
        if 'conv2d' in layer.name and '512' in str(layer.filters):
            bottleneck_layer = layer
            break
    
    if bottleneck_layer is None:
        # Fallback: use a layer around the middle
        bottleneck_layer = model.layers[len(model.layers)//2]
    
    encoder = Model(model.input, bottleneck_layer.output)
    return encoder.predict([binary_data, rgb_data])

def plot_dual_results(model, binary_test, rgb_test, target_test, num_samples=3):
    """
    Visualize dual-input reconstruction results
    """
    predictions = model.predict([binary_test[:num_samples], rgb_test[:num_samples]])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(num_samples):
        # Binary input (sum all channels)
        axes[i, 0].imshow(np.sum(binary_test[i], axis=-1), cmap='gray')
        axes[i, 0].set_title('Binary Input\n(All Channels)')
        axes[i, 0].axis('off')
        
        # Preprocessed RGB input
        axes[i, 1].imshow(rgb_test[i])
        axes[i, 1].set_title('RGB Input\n(With Gaps)')
        axes[i, 1].axis('off')
        
        # Target
        axes[i, 2].imshow(target_test[i])
        axes[i, 2].set_title('Target\n(Complete RGB)')
        axes[i, 2].axis('off')
        
        # Prediction
        axes[i, 3].imshow(predictions[i])
        axes[i, 3].set_title('Prediction\n(Reconstructed)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_metrics(history):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_model_dual(model, path):
    """Save trained dual-input model"""
    model.save(path)

def load_odysseynet_dual(path):
    """Load saved dual-input model"""
    custom_objects = {
        'combined_loss': combined_loss,
        'edge_loss': edge_loss
    }
    return tf.keras.models.load_model(path, custom_objects=custom_objects)

def example_usage_dual():
    """
    Example demonstrating how to use OdysseyNet Dual
    """
    print("OdysseyNet Dual - Dual-Input Autoencoder Demo")
    print("=" * 50)
    
    # Configuration
    BINARY_SHAPE = (128, 128, 5)
    RGB_SHAPE = (128, 128, 3)
    NUM_SAMPLES = 600
    EPOCHS = 50
    BATCH_SIZE = 16
    
    # Generate synthetic dual-input data
    print("Generating synthetic dual-input data...")
    binary_data, rgb_data, target_data = generate_dual_input_data(
        NUM_SAMPLES, BINARY_SHAPE, RGB_SHAPE
    )
    
    # Split data
    binary_train, binary_test, rgb_train, rgb_test, target_train, target_test = \
        train_test_split(binary_data, rgb_data, target_data, test_size=0.2, random_state=42)
    
    binary_train, binary_val, rgb_train, rgb_val, target_train, target_val = \
        train_test_split(binary_train, rgb_train, target_train, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(binary_train)}")
    print(f"Validation samples: {len(binary_val)}")
    print(f"Test samples: {len(binary_test)}")
    
    # Build model
    print("\nBuilding OdysseyNet Dual model...")
    model = build_odysseynet_dual(BINARY_SHAPE, RGB_SHAPE)
    print(f"Model parameters: {model.count_params():,}")
    
    # Train model
    print("\nTraining model...")
    history = train_odysseynet_dual(
        model, binary_train, rgb_train, target_train,
        binary_val, rgb_val, target_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    
    # Evaluate
    test_loss = model.evaluate([binary_test, rgb_test], target_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    
    # Extract features
    print("\nExtracting feature vectors...")
    features = extract_feature_vectors_dual(model, binary_test[:5], rgb_test[:5])
    print(f"Feature vector shape: {features.shape}")
    
    # Visualize results
    print("\nVisualizing results...")
    plot_training_metrics(history)
    plot_dual_results(model, binary_test, rgb_test, target_test)
    
    # Save model
    model_path = "odysseynet_dual_model.keras"
    save_model_dual(model, model_path)
    print(f"\nModel saved as: {model_path}")
    
    # Test loading
    loaded_model = load_odysseynet_dual(model_path)
    print("Model loaded successfully!")
    
    print("\nDemo complete!")

if __name__ == "__main__":
    example_usage_dual()