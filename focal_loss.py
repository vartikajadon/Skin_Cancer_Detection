import os
import logging

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import losses
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    losses = object  # Fallback for subclassing

logger = logging.getLogger(__name__)

class CategoricalFocalLoss(losses.Loss if TENSORFLOW_AVAILABLE else object):
    """
    Custom Focal Loss implementation for multi-class classification.
    Formulated to handle label smoothing and class balancing weights.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, name="categorical_focal_loss", **kwargs):
        if TENSORFLOW_AVAILABLE:
            super().__init__(name=name, **kwargs)
            self.alpha = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None
        else:
            self.alpha = alpha
            
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """
        Calculates loss between true one-hot targets and predicted softmax distributions.
        """
        if not TENSORFLOW_AVAILABLE:
            # Fallback mock for non-TF validation environments
            return 0.0

        # Apply label smoothing if configured and one-hot encoding is present
        if self.label_smoothing > 0.0:
            num_classes = tf.shape(y_pred)[-1]
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / tf.cast(num_classes, tf.float32))

        # Clip predictions to prevent log(0) numerical instability
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

        # Cross entropy loss component
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Apply Focal Loss focusing factor: (1 - p_pred)^gamma
        focal_factor = tf.math.pow(1.0 - y_pred, self.gamma)
        loss = focal_factor * cross_entropy

        # Apply class-wise weights (alpha) if configured
        if self.alpha is not None:
            loss = self.alpha * loss

        # Sum over classes, average over batch
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        """Allows serialization for model loading."""
        config = {
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing
        }
        if self.alpha is not None:
            # Convert alpha tensor to list for serialization
            config["alpha"] = self.alpha.numpy().tolist() if TENSORFLOW_AVAILABLE else self.alpha
            
        if TENSORFLOW_AVAILABLE:
            base_config = super().get_config()
            return {**base_config, **config}
        return config
