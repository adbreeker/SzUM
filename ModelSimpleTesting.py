import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import random
from keras_preprocessing.image import array_to_img
from DatasetPreparation import LoadDatasetWithNormalization

def get_random_image_from_dataset(test_dataset):
    """
    Get a random image from the dataset properly using LoadDatasetWithNormalization
    """
    # Get a batch from the dataset
    for batch_x, batch_y in test_dataset:
        # Get a random index from the batch
        idx = random.randint(0, batch_x.shape[0]-1)
        
        # Extract the image and its one-hot encoded label
        img = batch_x[idx]
        label = batch_y[idx]
        
        # Get the class name
        class_idx = np.argmax(label)
        class_names = list(test_dataset.class_indices.keys())
        class_name = class_names[class_idx]
        
        # Find the path of a representative original image (for display only)
        class_dir = os.path.join(test_dataset.directory, class_name)
        
        return img, class_name, class_dir
        
    # If we somehow didn't get an image
    raise ValueError("Could not get image from dataset")

class InteractivePlot:
    def __init__(self, model_path, test_dir, img_size=(128, 128)):
        self.model_path = model_path
        self.test_dir = test_dir
        self.img_size = img_size
        self.model = tf.keras.models.load_model(model_path)
        self.test_dataset = LoadDatasetWithNormalization(test_dir, target_size=img_size)
        self.class_names = list(self.test_dataset.class_indices.keys())
        
        # Create the figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(bottom=0.35)  # Increase from 0.25 to 0.35 to make much more room at bottom
        
        # Create the button - move it much lower
        self.button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])  # Adjusted position
        self.button = Button(self.button_ax, 'Try Another Image', color='lightblue', hovercolor='skyblue')
        self.button.on_clicked(self.update)
        
        # Draw the initial plot
        self.update(None)
        
    def update(self, event):
        # Clear axes for new plot
        self.ax1.clear()
        self.ax2.clear()
        
        # Get a random test image from the dataset
        img_normalized, true_class, class_dir = get_random_image_from_dataset(self.test_dataset)
        
        # Prepare the image for prediction (add batch dimension)
        img_for_prediction = np.expand_dims(img_normalized, axis=0)
        
        # Denormalize the image for display (multiply by 255 to reverse the 1./255 rescaling)
        img_denormalized = img_normalized * 255.0  # Denormalize
        
        # Ensure we have a rank-3 tensor
        if len(img_denormalized.shape) == 2:  # If it's a 2D array (height, width)
            # Add channel dimension to make it (height, width, 1)
            img_denormalized = np.expand_dims(img_denormalized, axis=-1)
        
        # Convert to uint8 before passing to array_to_img
        img_denormalized = img_denormalized.astype(np.uint8)
        
        # Convert to PIL image
        original_img = array_to_img(img_denormalized, scale=False)
        
        # Make prediction
        predictions = self.model.predict(img_for_prediction, verbose=0)
        predictions = predictions[0]  # Get the first (and only) prediction
        
        # Find predicted class
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[predicted_class_idx] * 100
        is_correct = (predicted_class == true_class)
        
        # Display the denormalized image (remove grayscale colormap for RGB)
        self.ax1.imshow(np.asarray(original_img))  # No cmap parameter for RGB images
        self.ax1.set_title(f"True class: {true_class}")
        self.ax1.axis('off')
        
        # Display prediction confidence as a bar chart
        y_pos = np.arange(len(self.class_names))
        bars = self.ax2.barh(y_pos, predictions, align='center')
        self.ax2.set_yticks(y_pos)
        self.ax2.set_yticklabels(self.class_names)
        self.ax2.set_xlabel('Confidence')
        self.ax2.set_title('Model Predictions')
        
        # Set the highest confidence prediction color to green
        for i, bar in enumerate(bars):
            if i == predicted_class_idx:
                bar.set_color('green' if is_correct else 'red')
        
        # Add text annotation with confidence percentages - UPDATED WITH MORE PRECISION
        for i, v in enumerate(predictions):
            # Format with 4 decimal places (instead of 1) for better precision
            self.ax2.text(v + 0.01, i, f"{v:.4f}", va='center')
        
        # Display conclusion text
        conclusion = f"I am {confidence:.1f}% sure this is a {predicted_class}"
        result_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        
        # Remove old text annotations if they exist
        if hasattr(self, 'conclusion_text'):
            self.conclusion_text.remove()
        if hasattr(self, 'result_text'):
            self.result_text.remove()
        
        # Move the conclusion and result text down, but keep them above the button
        self.conclusion_text = self.fig.text(0.5, 0.22, conclusion, ha='center', fontsize=14,
                      bbox={'facecolor':'lightgreen' if is_correct else 'salmon', 'alpha':0.5, 'pad':5})
        
        self.result_text = self.fig.text(0.5, 0.15, result_text, ha='center', fontsize=16,
                      color='green' if is_correct else 'red', weight='bold')
        
        # Adjust the layout to accommodate the new text positions
        self.fig.tight_layout(rect=[0, 0.35, 1, 0.95])  # Increase bottom margin to match subplots_adjust
        self.fig.canvas.draw_idle()
        
        # Print confidence values in console with more precision
        print(f"\nPrediction confidence for each class:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: {predictions[i]:.4f}")  # Raw values with more precision

        # Display conclusion in console
        print(f"\n{conclusion}")
        print(f"Prediction is {result_text}")

def predict_and_visualize(model_path, test_dir, img_size=(128, 128)):
    """
    Load model, select a random test image, make prediction and visualize results with an interactive button
    """
    # Create interactive plot
    interactive_plot = InteractivePlot(model_path, test_dir, img_size)
    plt.show()

if __name__ == "__main__":
    # Configure parameters
    MODEL_PATH = "./TrainingOutputs/Split2/MODEL2.h5"
    TEST_DIR = "./Split2/test"
    IMG_SIZE = (128, 128)
    
    # Run prediction and visualization with interactive button
    predict_and_visualize(MODEL_PATH, TEST_DIR, IMG_SIZE)