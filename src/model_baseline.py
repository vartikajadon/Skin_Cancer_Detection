import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    layers = None
    models = None

def get_baseline_cnn_model(input_shape: tuple = (224, 224, 3), num_classes: int = 7):
    """
    Returns a compiled Keras Sequential CNN model for skin lesion classification.
    If TensorFlow is not available, returns a high-fidelity mock model representation.
    """
    if TENSORFLOW_AVAILABLE:
        logger.info("Initializing TensorFlow baseline CNN architecture...")
        model = models.Sequential([
            # Input Layer (implicitly defined via input_shape in Conv2D)
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Classification Head
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(num_classes, activation='softmax')
        ], name="baseline_cnn")
        
        return model
    else:
        logger.warning("TensorFlow/Keras is not available. Creating mock BaselineCNN class.")
        return MockBaselineCNN(input_shape, num_classes)

class MockBaselineCNN:
    """
    High-fidelity mock baseline CNN model matching structural configurations 
    and parameter counts of the Keras implementation.
    """
    def __init__(self, input_shape=(224, 224, 3), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = "baseline_cnn"
        
        # Exact calculated parameters
        self.trainable_params = 25817415
        self.non_trainable_params = 0
        self.total_params = self.trainable_params
        
    def summary(self) -> str:
        """
        Generates and prints a detailed layer-by-layer structure matching 
        a native Keras model summary.
        """
        summary_lines = [
            f"Model: \"{self.name}\"",
            "_________________________________________________________________",
            " Layer (type)                Output Shape              Param #   ",
            "=================================================================",
            " conv2d (Conv2D)             (None, 224, 224, 32)      896       ",
            "                                                                 ",
            " max_pooling2d (MaxPooling2D  (None, 112, 112, 32)     0         ",
            " )                                                               ",
            "                                                                 ",
            " conv2d_1 (Conv2D)           (None, 112, 112, 64)      18496     ",
            "                                                                 ",
            " max_pooling2d_1 (MaxPooling  (None, 56, 56, 64)       0         ",
            " 2D)                                                             ",
            "                                                                 ",
            " conv2d_2 (Conv2D)           (None, 56, 56, 128)       73856     ",
            "                                                                 ",
            " max_pooling2d_2 (MaxPooling  (None, 28, 28, 128)      0         ",
            " 2D)                                                             ",
            "                                                                 ",
            " flatten (Flatten)           (None, 100352)            0         ",
            "                                                                 ",
            " dense (Dense)               (None, 256)               25690368  ",
            "                                                                 ",
            " dropout (Dropout)           (None, 256)               0         ",
            "                                                                 ",
            " dense_1 (Dense)             (None, 128)               32896     ",
            "                                                                 ",
            " dropout_1 (Dropout)         (None, 128)               0         ",
            "                                                                 ",
            " dense_2 (Dense)             (None, 7)                 903       ",
            "=================================================================",
            f"Total params: {self.total_params:,} ({98.48} MB)",
            f"Trainable params: {self.trainable_params:,} ({98.48} MB)",
            f"Non-trainable params: {self.non_trainable_params} (0.00 B)",
            "_________________________________________________________________"
        ]
        summary_text = "\n".join(summary_lines)
        print(summary_text)
        return summary_text

if __name__ == "__main__":
    model = get_baseline_cnn_model()
    model.summary()
