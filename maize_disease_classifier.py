# Improved Multiclass Maize Disease Classifier using EfficientNet
# Supports Healthy, Gray Leaf Spot, Blight, and Common Rust detection
# Complete implementation for Google Colab

# 1. Install required packages
!pip install tensorflow==2.12.0
!pip install opencv-python
!pip install scikit-learn
!pip install seaborn

# 2. Import required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from google.colab import files, drive
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization,
    Multiply, Reshape, Permute, multiply, Lambda, Conv2D, Add, UpSampling2D,
    Concatenate, Activation, LeakyReLU
)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import backend as K
import cv2
import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import math

# 3. Mount Google Drive
drive.mount('/content/drive')

# 4. Define all helper functions
def create_efficientnet_model(num_classes=4, efficientnet_version='B3', fine_tune_layers=30):
    """
    Creates an improved EfficientNet model with enhanced attention mechanism and FPN
    """
    # Choose EfficientNet version
    if efficientnet_version == 'B3':
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
        img_size = (300, 300)
    else:
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_size = (224, 224)

    # Freeze most layers and unfreeze top layers for fine-tuning
    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # Build model with enhanced attention and FPN
    inputs = Input(shape=img_size + (3,))
    
    # Feature Pyramid Network (FPN)
    def build_fpn_block(x, filters):
        x = Conv2D(filters, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x

    # Get intermediate features from base model
    x = inputs
    features = []
    skip_connections = {}
    
    # Process through base model and collect features
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Multiply) and 'se_excite' in layer.name:
            # Skip SE blocks as they require special handling
            continue
        elif isinstance(layer, tf.keras.layers.Add):
            # Store skip connection
            if len(layer.input) == 2:
                skip_connections[layer.name] = layer.input[1]
            x = layer(x)
        else:
            x = layer(x)
            
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.filters >= 256:
            features.append(x)

    # Build FPN with proper upsampling
    fpn_features = []
    for i, feature in enumerate(features):
        fpn_feature = build_fpn_block(feature, 256)
        if i > 0:
            # Calculate the target size for upsampling
            target_shape = K.int_shape(fpn_features[-1])[1:3]
            current_shape = K.int_shape(fpn_feature)[1:3]
            
            # Calculate the required upsampling factor
            scale_factor = (target_shape[0] // current_shape[0], 
                          target_shape[1] // current_shape[1])
            
            # Apply upsampling
            if scale_factor[0] > 1 or scale_factor[1] > 1:
                fpn_feature = UpSampling2D(size=scale_factor)(fpn_feature)
            
            # Ensure shapes match before adding
            if K.int_shape(fpn_feature)[1:3] != target_shape:
                fpn_feature = tf.keras.layers.Lambda(
                    lambda x: tf.image.resize(x, target_shape, method='bilinear')
                )(fpn_feature)
            
            fpn_feature = Add()([fpn_feature, fpn_features[-1]])
        fpn_features.append(fpn_feature)

    # Enhanced attention mechanism
    def attention_block(x, g, filters):
        theta_x = Conv2D(filters, (1, 1), strides=(1, 1))(x)
        phi_g = Conv2D(filters, (1, 1), strides=(1, 1))(g)
        f = Activation('relu')(Add()([theta_x, phi_g]))
        psi_f = Conv2D(1, (1, 1), strides=(1, 1))(f)
        rate = Activation('sigmoid')(psi_f)
        return Multiply()([x, rate])

    # Apply attention to FPN features
    attended_features = []
    for i, feature in enumerate(fpn_features):
        if i > 0:
            attended = attention_block(feature, fpn_features[i-1], 256)
            attended_features.append(attended)
        else:
            attended_features.append(feature)

    # Combine features with proper resizing
    target_size = K.int_shape(attended_features[0])[1:3]
    resized_features = []
    for feature in attended_features:
        if K.int_shape(feature)[1:3] != target_size:
            feature = tf.keras.layers.Lambda(
                lambda x: tf.image.resize(x, target_size, method='bilinear')
            )(feature)
        resized_features.append(feature)
    
    x = Concatenate()(resized_features)
    x = GlobalAveragePooling2D()(x)
    
    # Classification head with residual connections
    def residual_block(x, filters):
        input_filters = K.int_shape(x)[-1]
        y = Dense(filters)(x)
        y = BatchNormalization()(y)
        y = LeakyReLU(0.2)(y)
        y = Dense(filters)(y)
        y = BatchNormalization()(y)
        y = LeakyReLU(0.2)(y)
        
        # Project input if dimensions don't match
        if input_filters != filters:
            x = Dense(filters)(x)
            x = BatchNormalization()(x)
        
        return Add()([x, y])

    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    
    # Add residual blocks
    x = residual_block(x, 1024)
    x = residual_block(x, 512)
    
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create model with proper input/output connections
    model = Model(inputs=inputs, outputs=outputs, name='efficientnet_disease_classifier')

    # Compile with label smoothing
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model, img_size

def random_erasing(img, probability=0.3, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3):
    """
    Random erasing augmentation
    """
    if np.random.random() > probability:
        return img
    
    h, w, c = img.shape
    area = h * w
    
    for _ in range(100):
        erase_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, r2)
        
        h_erase = int(np.sqrt(erase_area * aspect_ratio))
        w_erase = int(np.sqrt(erase_area / aspect_ratio))
        
        if h_erase < h and w_erase < w:
            i = np.random.randint(0, h - h_erase)
            j = np.random.randint(0, w - w_erase)
            img[i:i+h_erase, j:j+w_erase, :] = np.random.uniform(0, 1, (h_erase, w_erase, c))
            return img
    
    return img

def mixup(x, y, alpha=0.2):
    """
    Mixup augmentation
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))

    mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, index)

    return mixed_x, mixed_y

class MixupGenerator:
    """
    Generator that applies mixup augmentation
    """
    def __init__(self, generator, alpha=0.2):
        self.generator = generator
        self.alpha = alpha

    def __iter__(self):
        return self

    def __next__(self):
        x, y = next(self.generator)
        return mixup(x, y, self.alpha)

def cosine_annealing_with_warmup(epoch, lr, warmup_epochs=5, total_epochs=30):
    """
    Learning rate schedule with warmup and cosine annealing
    """
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

def prepare_data_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=16):
    """
    Prepares data generators with enhanced augmentation for disease detection
    """
    # Enhanced augmentation for better disease feature preservation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.0,
        channel_shift_range=50.0,
        preprocessing_function=lambda x: random_erasing(x, probability=0.3)
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = valid_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = valid_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    class_names = [k for k, v in sorted(train_generator.class_indices.items(), key=lambda item: item[1])]
    print(f"Class names: {class_names}")

    return train_generator, val_generator, test_generator, class_names

def plot_training_history(history, save_path):
    """
    Plot training history
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))

    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the raw counts
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Counts)')

    # Plot the percentages
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 5. Main execution block
def main():
    # Define paths
    BASE_DIR = '/content/drive/MyDrive/Maize disease'
    HEALTHY_DIR = os.path.join(BASE_DIR, 'Healthy_leaf')
    GLS_DIR = os.path.join(BASE_DIR, 'Gray_leaf_spot')
    BLIGHT_DIR = os.path.join(BASE_DIR, 'Blight')
    RUST_DIR = os.path.join(BASE_DIR, 'Common_Rust')
    DATASET_DIR = os.path.join(BASE_DIR, 'maize_dataset_multiclass')
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')

    # Create necessary directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Model configuration
    EFFICIENTNET_VERSION = 'B3'
    IMG_SIZE = (300, 300)
    FINE_TUNE_LAYERS = 30
    BATCH_SIZE = 8
    EPOCHS = 30
    RUST_WEIGHT_MULTIPLIER = 1.5

    # Create and compile model
    print("Creating model...")
    model, img_size = create_efficientnet_model(
        num_classes=4,
        efficientnet_version=EFFICIENTNET_VERSION,
        fine_tune_layers=FINE_TUNE_LAYERS
    )
    model.summary()

    # Prepare data generators
    print("Preparing data generators...")
    train_generator, val_generator, test_generator, class_names = prepare_data_generators(
        os.path.join(DATASET_DIR, 'train'),
        os.path.join(DATASET_DIR, 'val'),
        os.path.join(DATASET_DIR, 'test'),
        img_size=img_size,
        batch_size=BATCH_SIZE
    )

    # Apply mixup to training generator
    train_generator = MixupGenerator(train_generator, alpha=0.2)

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.generator.classes),
        y=train_generator.generator.classes
    )
    class_weight_dict = dict(zip(np.unique(train_generator.generator.classes), class_weights))

    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_SAVE_DIR, 'efficientnet_disease_model_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    lr_scheduler = LearningRateScheduler(
        lambda epoch: cosine_annealing_with_warmup(
            epoch,
            lr=0.00005,
            warmup_epochs=5,
            total_epochs=EPOCHS
        )
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.000005,
        verbose=1
    )

    callbacks = [checkpoint, early_stopping, reduce_lr, lr_scheduler]

    # Train the model
    print("Training model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.generator.samples // train_generator.generator.batch_size,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    # Save final model
    model.save(os.path.join(MODEL_SAVE_DIR, 'efficientnet_disease_model_final.h5'))

    # Plot training history
    plot_training_history(history, os.path.join(MODEL_SAVE_DIR, 'training_history.png'))

    # Evaluate on test set
    print("Evaluating model...")
    results = model.evaluate(test_generator)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")

    # Get predictions for confusion matrix
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, class_names, 
                         os.path.join(MODEL_SAVE_DIR, 'confusion_matrix.png'))

    # Save class names
    with open(os.path.join(MODEL_SAVE_DIR, 'disease_class_names.txt'), 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    print("Training and evaluation complete!")
    print(f"Model saved to {os.path.join(MODEL_SAVE_DIR, 'efficientnet_disease_model_final.h5')}")

if __name__ == "__main__":
    main() 