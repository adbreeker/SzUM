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
    for batch_x, batch_y in test_dataset:
        idx = random.randint(0, batch_x.shape[0]-1)
        
        img = batch_x[idx]
        label = batch_y[idx]
        
        class_idx = np.argmax(label)
        class_names = list(test_dataset.class_indices.keys())
        class_name = class_names[class_idx]
        
        class_dir = os.path.join(test_dataset.directory, class_name)
        
        return img, class_name, class_dir
        
    raise ValueError("Could not get image from dataset")

class InteractivePlot:
    def __init__(self, model_path, test_dir, img_size=(128, 128)):
        self.model_path = model_path
        self.test_dir = test_dir
        self.img_size = img_size
        self.model = tf.keras.models.load_model(model_path)
        self.test_dataset = LoadDatasetWithNormalization(test_dir, target_size=img_size)
        self.class_names = list(self.test_dataset.class_indices.keys())
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.canvas.manager.set_window_title("Model Prediction Viewer")
        plt.subplots_adjust(bottom=0.35)
        
        self.button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.button = Button(self.button_ax, 'Try Another Image', color='lightblue', hovercolor='skyblue')
        self.button.on_clicked(self.update)
        
        self.update(None)
        
    def update(self, event):
        self.ax1.clear()
        self.ax2.clear()
        
        img_normalized, true_class, class_dir = get_random_image_from_dataset(self.test_dataset)
        
        img_for_prediction = np.expand_dims(img_normalized, axis=0)
        
        img_denormalized = img_normalized * 255.0
        
        if len(img_denormalized.shape) == 2:
            img_denormalized = np.expand_dims(img_denormalized, axis=-1)
        
        img_denormalized = img_denormalized.astype(np.uint8)
        
        original_img = array_to_img(img_denormalized, scale=False)
        
        predictions = self.model.predict(img_for_prediction, verbose=0)
        predictions = predictions[0]
        
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[predicted_class_idx] * 100
        is_correct = (predicted_class == true_class)
        
        self.ax1.imshow(np.asarray(original_img)) 
        self.ax1.set_title(f"True class: {true_class}")
        self.ax1.axis('off')
        
        y_pos = np.arange(len(self.class_names))
        bars = self.ax2.barh(y_pos, predictions, align='center')
        self.ax2.set_yticks(y_pos)
        self.ax2.set_yticklabels(self.class_names)
        self.ax2.set_xlabel('Confidence')
        self.ax2.set_title(f'Model({MODEL_PATH}) Predictions')
        
        for i, bar in enumerate(bars):
            if i == predicted_class_idx:
                bar.set_color('green' if is_correct else 'red')
        
        for i, v in enumerate(predictions):
            self.ax2.text(v + 0.01, i, f"{v:.4f}", va='center')
        
        conclusion = f"I am {confidence:.1f}% sure this is a {predicted_class}"
        result_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        
        if hasattr(self, 'conclusion_text'):
            self.conclusion_text.remove()
        if hasattr(self, 'result_text'):
            self.result_text.remove()
        
        self.conclusion_text = self.fig.text(0.5, 0.22, conclusion, ha='center', fontsize=14,
                      bbox={'facecolor':'lightgreen' if is_correct else 'salmon', 'alpha':0.5, 'pad':5})
        
        self.result_text = self.fig.text(0.5, 0.15, result_text, ha='center', fontsize=16,
                      color='green' if is_correct else 'red', weight='bold')
        
        self.fig.tight_layout(rect=[0, 0.35, 1, 0.95])
        self.fig.canvas.draw_idle()
        
        print(f"\nPrediction confidence for each class:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: {predictions[i]:.4f}")

        print(f"\n{conclusion}")
        print(f"Prediction is {result_text}")

def predict_and_visualize(model_path, test_dir, img_size=(128, 128)):
    interactive_plot = InteractivePlot(model_path, test_dir, img_size)
    plt.show()

# ------------------------------------------------------------- RUN -------------------------------------------------------

if __name__ == "__main__":
    splits = {
        "1": ("./TrainingOutputs/Split1/MODEL1.keras", "./Split1/test"),
        "2": ("./TrainingOutputs/Split2/MODEL2.keras", "./Split2/test"),
        "3": ("./TrainingOutputs/Split3/MODEL3.keras", "./Split3/test"),
    }
    print("Choose split model to use:")
    for k in splits:
        print(f"{k}: {splits[k][0]}")
    choice = input("Enter split number (1/2/3): ").strip()
    if choice not in splits:
        print("Invalid choice. Using split 1 by default.")
        choice = "1"
    MODEL_PATH, TEST_DIR = splits[choice]
    IMG_SIZE = (128, 128)
    interactive_plot = InteractivePlot(MODEL_PATH, TEST_DIR, IMG_SIZE)

    plt.show()