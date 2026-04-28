import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import shutil
from pathlib import Path
import random
from sklearn.metrics import confusion_matrix, classification_report

# Constants for GLS model parameters
TEMP_THRESHOLD_LOW = 15
TEMP_THRESHOLD_OPTIMAL_LOW = 20
TEMP_THRESHOLD_OPTIMAL_HIGH = 28
TEMP_THRESHOLD_HIGH = 32

RH_THRESHOLD_LOW = 70
RH_THRESHOLD_OPTIMAL_LOW = 86
RH_THRESHOLD_OPTIMAL_HIGH = 96

LW_THRESHOLD = 5  # Leaf wetness threshold in hours

RISK_VALUES = {
    "High": 2,
    "Medium": 1,
    "Low": 0.5,
    "No Risk": 0
}

def prepare_dataset(healthy_dir, gls_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split the original healthy and GLS datasets into train, validation, and test sets.
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'validation', 'test']:
        for label in ['healthy', 'gray_leaf_spot']:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

    # Function to process and split one class of images
    def process_class(src_dir, class_name):
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(src_dir).glob(ext))
            image_files.extend(Path(src_dir).glob("**/" + ext))
        
        image_files = [str(file) for file in image_files]
        print(f"Found {len(image_files)} {class_name} images")

        # Shuffle files
        random.seed(seed)
        random.shuffle(image_files)

        # Calculate split indices
        n_total = len(image_files)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]

        # Copy files to their respective directories
        target_class = 'healthy' if class_name == 'healthy' else 'gray_leaf_spot'

        # Copy training files
        print(f"Copying {len(train_files)} {class_name} images to training set...")
        for file in train_files:
            dest_path = os.path.join(output_dir, 'train', target_class, os.path.basename(file))
            shutil.copy2(file, dest_path)

        # Copy validation files
        print(f"Copying {len(val_files)} {class_name} images to validation set...")
        for file in val_files:
            dest_path = os.path.join(output_dir, 'validation', target_class, os.path.basename(file))
            shutil.copy2(file, dest_path)

        # Copy test files
        print(f"Copying {len(test_files)} {class_name} images to test set...")
        for file in test_files:
            dest_path = os.path.join(output_dir, 'test', target_class, os.path.basename(file))
            shutil.copy2(file, dest_path)

    # Process both classes
    process_class(healthy_dir, 'healthy')
    process_class(gls_dir, 'gls')

    print("\nDataset preparation complete!")
    print(f"Train, validation, and test sets created in {output_dir}")

    # Print statistics
    for split in ['train', 'validation', 'test']:
        healthy_count = len(os.listdir(os.path.join(output_dir, split, 'healthy')))
        gls_count = len(os.listdir(os.path.join(output_dir, split, 'gray_leaf_spot')))
        total = healthy_count + gls_count
        print(f"{split.capitalize()} set: {total} images (Healthy: {healthy_count}, GLS: {gls_count})")

def build_gls_classification_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build a CNN model for Gray Leaf Spot disease classification.
    """
    # Use MobileNetV2 as base model (efficient for deployment)
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # Freeze base model layers
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model

def train_gls_model(train_dir, validation_dir, batch_size=32, epochs=10, model_save_path='models/gls_model.h5'):
    """
    Train the GLS detection model using image data from the specified directories.
    """
    # Create directory for saving models
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    input_shape = (224, 224, 3)

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Build model
    model = build_gls_classification_model(input_shape=input_shape, num_classes=len(train_generator.class_indices))

    # Callbacks
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )

    # Plot training history
    plot_training_history(history)

    return model, history

def plot_training_history(history):
    """Plot training and validation metrics"""
    # Create directory for results
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.show()

def evaluate_model(model, test_dir, batch_size=32):
    """
    Evaluate the trained model on the test set.
    """
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate the model
    results = model.evaluate(test_generator)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")

    # Predict on test set
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true classes
    true_classes = test_generator.classes

    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    class_names = list(test_generator.class_indices.keys())
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('results/confusion_matrix.png')
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

    return results, cm

class GLSRiskAssessor:
    """Implements the GLS risk assessment logic based on temperature, humidity, and leaf wetness"""

    def calculate_daily_risk(self, temp_hourly, rh_hourly, lw_hourly):
        """
        Calculate the daily risk based on the GLS model parameters
        """
        # Count hours meeting different conditions
        high_risk_hours = 0
        medium_risk_hours = 0
        low_risk_hours = 0

        # Check each hour for risk conditions
        for hour in range(24):
            temp = temp_hourly[hour]
            rh = rh_hourly[hour]

            # High risk condition
            if (TEMP_THRESHOLD_OPTIMAL_LOW <= temp <= TEMP_THRESHOLD_OPTIMAL_HIGH and
                RH_THRESHOLD_OPTIMAL_LOW <= rh <= RH_THRESHOLD_OPTIMAL_HIGH):
                high_risk_hours += 1

            # Medium risk condition
            elif (TEMP_THRESHOLD_LOW <= temp <= TEMP_THRESHOLD_HIGH and
                  RH_THRESHOLD_LOW <= rh < RH_THRESHOLD_OPTIMAL_LOW):
                medium_risk_hours += 1

            # Low risk condition
            elif (TEMP_THRESHOLD_OPTIMAL_LOW <= temp <= TEMP_THRESHOLD_OPTIMAL_HIGH and
                  rh < RH_THRESHOLD_LOW):
                low_risk_hours += 1

        # Determine initial risk based on duration thresholds
        initial_risk = "No Risk"
        if high_risk_hours >= 12:
            initial_risk = "High"
        elif medium_risk_hours >= 12:
            initial_risk = "Medium"
        elif low_risk_hours >= 8:
            initial_risk = "Low"

        # If risk is High or Medium, adjust based on leaf wetness
        if initial_risk in ["High", "Medium"]:
            # Count hours with leaf wetness > 1%
            lw_hours = sum(1 for lw in lw_hourly if lw >= 1)

            if initial_risk == "High":
                if lw_hours > LW_THRESHOLD:
                    final_risk = "High"
                else:
                    final_risk = "Medium"
            else:  # Medium risk
                if lw_hours > LW_THRESHOLD:
                    final_risk = "Medium"
                else:
                    final_risk = "Low"
        else:
            final_risk = initial_risk

        return final_risk, RISK_VALUES[final_risk]

    def calculate_spray_recommendation(self, risk_history):
        """
        Determine if spraying is recommended based on risk history
        """
        if not risk_history:
            return False, None

        # Sort history by date
        sorted_history = sorted(risk_history, key=lambda x: x[0])
        last_date = sorted_history[-1][0]

        # Check 7-day high risk condition
        seven_day_history = [entry for entry in sorted_history
                           if (last_date - entry[0]).days < 7]
        seven_day_sum = sum(entry[2] for entry in seven_day_history)

        if seven_day_sum >= 9:
            return True, "7-day"

        # Check 10-day medium risk condition
        ten_day_history = [entry for entry in sorted_history
                         if (last_date - entry[0]).days < 10]
        ten_day_sum = sum(entry[2] for entry in ten_day_history)

        if ten_day_sum >= 8:
            return True, "10-day"

        # Check 14-day low risk condition
        fourteen_day_history = [entry for entry in sorted_history
                              if (last_date - entry[0]).days < 14]
        fourteen_day_sum = sum(entry[2] for entry in fourteen_day_history)

        if fourteen_day_sum >= 10:
            return True, "14-day"

        return False, None

def plot_risk_history(risk_history):
    """Plot the risk history"""
    # Sort by date
    sorted_history = sorted(risk_history, key=lambda x: x[0])
    
    # Extract dates and risk values
    dates = [entry[0] for entry in sorted_history]
    risk_values = [entry[2] for entry in sorted_history]
    risk_levels = [entry[1] for entry in sorted_history]
    
    # Create colors based on risk level
    colors = []
    for level in risk_levels:
        if level == "High":
            colors.append('red')
        elif level == "Medium":
            colors.append('orange')
        elif level == "Low":
            colors.append('yellow')
        else:
            colors.append('green')
    
    plt.figure(figsize=(10, 5))
    plt.bar(dates, risk_values, color=colors)
    plt.xlabel('Date')
    plt.ylabel('Risk Value')
    plt.title('GLS Risk History')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Create a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='High Risk'),
        Patch(facecolor='orange', label='Medium Risk'),
        Patch(facecolor='yellow', label='Low Risk'),
        Patch(facecolor='green', label='No Risk')
    ]
    plt.legend(handles=legend_elements)
    
    plt.savefig('results/risk_history.png')
    plt.show()

def predict_on_image(model, image_path):
    """
    Make a prediction for a single image
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Map class index to label
    class_labels = ['healthy', 'gray_leaf_spot']
    predicted_class = class_labels[predicted_class_idx]
    
    # Display the results
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f'Prediction: {predicted_class.capitalize()} (Confidence: {confidence:.2f})')
    plt.axis('off')
    plt.savefig('results/prediction_result.png')
    plt.show()
    
    return predicted_class, confidence

def main():
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("Gray Leaf Spot Disease Detection and Management System")
    print("=====================================================")
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Prepare dataset and train a new model")
    print("2. Use existing model for prediction")
    print("3. Run risk assessment only")
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        # Set paths for your image folders - CHANGE THESE TO YOUR ACTUAL PATHS!
        healthy_dir = input("\nEnter path to folder with healthy leaf images: ")
        gls_dir = input("Enter path to folder with Gray Leaf Spot images: ")
        processed_data_dir = "processed_data"
        
        # Prepare dataset
        print("\n1. Preparing Dataset")
        print("------------------")
        prepare_dataset(
            healthy_dir=healthy_dir,
            gls_dir=gls_dir,
            output_dir=processed_data_dir
        )
        
        # Train model
        print("\n2. Training the GLS Detection Model")
        print("-----------------------------------")
        train_dir = os.path.join(processed_data_dir, "train")
        validation_dir = os.path.join(processed_data_dir, "validation")
        test_dir = os.path.join(processed_data_dir, "test")
        
        epochs = int(input("Number of training epochs (recommended 10-20): ") or "10")
        
        model, history = train_gls_model(
            train_dir=train_dir,
            validation_dir=validation_dir,
            batch_size=32,
            epochs=epochs,
            model_save_path='models/gls_model.h5'
        )
        
        # Evaluate model
        print("\n3. Evaluating Model Performance")
        print("------------------------------")
        evaluate_model(model, test_dir)
        
    elif choice == '2':
        # Load existing model
        model_path = input("\nEnter path to trained model file (default: models/gls_model.h5): ") or "models/gls_model.h5"
        
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            
            # Make prediction on a new image
            image_path = input("\nEnter path to leaf image for prediction: ")
            
            if os.path.exists(image_path):
                predicted_class, confidence = predict_on_image(model, image_path)
                print(f"\nPrediction: {predicted_class}")
                print(f"Confidence: {confidence:.4f}")
                
                if predicted_class == 'gray_leaf_spot':
                    print("\nRecommendation:")
                    print("- This leaf shows signs of Gray Leaf Spot disease")
                    print("- Consider treatment with fungicide if disease is widespread")
                    print("- Monitor environmental conditions for disease progression risk")
                else:
                    print("\nRecommendation:")
                    print("- This leaf appears healthy")
                    print("- Continue regular monitoring for early signs of disease")
            else:
                print(f"Error: Image file {image_path} does not exist")
        else:
            print(f"Error: Model file {model_path} does not exist")
            
    elif choice == '3':
        # Run risk assessment
        print("\nGray Leaf Spot Risk Assessment")
        print("------------------------------")
        
        risk_assessor = GLSRiskAssessor()
        
        # Ask for weather data
        use_default = input("\nUse default weather data? (y/n): ").lower() in ['y', 'yes', '']
        
        if use_default:
            # Default data
            temp_data = [25] * 24  # 25°C for 24 hours
            rh_data = [90] * 12 + [80] * 12  # 90% RH for 12 hours, 80% for 12 hours
            lw_data = [10] * 6 + [0] * 18  # Leaf wetness for 6 hours
        else:
            print("\nPlease enter hourly temperature data (24 values between 0-40°C):")
            print("For simplicity, you can enter one value to use for all 24 hours.")
            temp_input = input("Temperature (°C): ")
            if ',' in temp_input:
                temp_data = [float(t.strip()) for t in temp_input.split(',')]
                if len(temp_data) < 24:
                    temp_data = temp_data + [temp_data[-1]] * (24 - len(temp_data))
                temp_data = temp_data[:24]
            else:
                temp_data = [float(temp_input)] * 24
            
            print("\nPlease enter hourly relative humidity data (24 values between 0-100%):")
            print("For simplicity, you can enter one value to use for all 24 hours.")
            rh_input = input("Relative Humidity (%): ")
            if ',' in rh_input:
                rh_data = [float(rh.strip()) for rh in rh_input.split(',')]
                if len(rh_data) < 24:
                    rh_data = rh_data + [rh_data[-1]] * (24 - len(rh_data))
                rh_data = rh_data[:24]
            else:
                rh_data = [float(rh_input)] * 24
            
            print("\nPlease enter hourly leaf wetness data (24 values, % of leaf wetness):")
            print("For simplicity, you can enter one value for wet hours and one for dry hours.")
            wet_hours = int(input("Number of wet hours (0-24): "))
            if wet_hours > 24:
                wet_hours = 24
            lw_data = [10] * wet_hours + [0] * (24 - wet_hours)
        
        # Calculate daily risk
        risk_level, risk_value = risk_assessor.calculate_daily_risk(temp_data, rh_data, lw_data)
        print(f"\nDaily Risk Assessment: {risk_level} (Value: {risk_value})")
        
        # Generate risk history
        use_simulated = input("\nGenerate simulated risk history? (y/n): ").lower() in ['y', 'yes', '']
        
        if use_simulated:
            today = datetime.now().date()
            
            # Sample risk pattern
            risk_pattern = [
                ("High", 2),
                ("High", 2),
                ("Medium", 1),
                ("Low", 0.5),
                ("Medium", 1),
                ("High", 2),
                (risk_level, risk_value)
            ]
            
            # Create risk history for the past week
            risk_history = []
            for i in range(7):
                day = today - timedelta(days=6-i)
                pattern_idx = i % len(risk_pattern)
                level, value = risk_pattern[pattern_idx]
                risk_history.append((day, level, value))
            
            # Plot risk history
            plot_risk_history(risk_history)
            
            # Calculate spray recommendation
            spray_recommended, recommendation_type = risk_assessor.calculate_spray_recommendation(risk_history)
            print(f"\nSpray Recommendation: {'Yes' if spray_recommended else 'No'}")
            if spray_recommended:
                print(f"Recommendation Type: {recommendation_type}")
                if recommendation_type == "7-day":
                    print("  Explanation: High risk conditions detected in the past 7 days")
                elif recommendation_type == "10-day":
                    print("  Explanation: Medium risk conditions detected in the past 10 days")
                elif recommendation_type == "14-day":
                    print("  Explanation: Low risk conditions detected in the past 14 days")
            
            print("\nRecommended Actions:")
            if spray_recommended:
                print("1. Apply fungicide within the next 24-48 hours")
                print("2. Use a systemic fungicide with protective and curative properties")
                print("3. Ensure proper coverage of the entire plant, especially the lower leaves")
                print("4. Monitor weather conditions and avoid spraying if rain is expected within 24 hours")
            else:
                print("1. Continue regular monitoring of weather conditions")
                print("2. Scout fields for early symptoms of Gray Leaf Spot")
                print("3. Prepare spray equipment for potential application in the near future")
                print("4. Review your fungicide inventory to ensure supplies are available if needed")
    
    else:
        print("Invalid choice. Please run the program again.")
    
    print("\nProgram Completed!")

if __name__ == "__main__":
    main()