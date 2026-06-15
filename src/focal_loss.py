# Copy of focal_loss.py to src folder
from pathlib import Path
import sys

# Import from parent directory if possible, or duplicate definition
import os
try:
    from focal_loss import CategoricalFocalLoss
except ImportError:
    # Inline duplicate definition to ensure robust stand-alone import
    try:
        import tensorflow as tf
        from tensorflow.keras import losses
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        tf = None
        losses = object

    class CategoricalFocalLoss(losses.Loss if TENSORFLOW_AVAILABLE else object):
        def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, name="categorical_focal_loss", **kwargs):
            if TENSORFLOW_AVAILABLE:
                super().__init__(name=name, **kwargs)
                self.alpha = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None
            else:
                self.alpha = alpha
            self.gamma = gamma
            self.label_smoothing = label_smoothing

        def call(self, y_true, y_pred):
            if not TENSORFLOW_AVAILABLE:
                return 0.0
            if self.label_smoothing > 0.0:
                num_classes = tf.shape(y_pred)[-1]
                y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / tf.cast(num_classes, tf.float32))
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
            cross_entropy = -y_true * tf.math.log(y_pred)
            focal_factor = tf.math.pow(1.0 - y_pred, self.gamma)
            loss = focal_factor * cross_entropy
            if self.alpha is not None:
                loss = self.alpha * loss
            return tf.reduce_sum(loss, axis=-1)

        def get_config(self):
            config = {"gamma": self.gamma, "label_smoothing": self.label_smoothing}
            if self.alpha is not None:
                config["alpha"] = self.alpha.numpy().tolist() if TENSORFLOW_AVAILABLE else self.alpha
            if TENSORFLOW_AVAILABLE:
                base_config = super().get_config()
                return {**base_config, **config}
            return config
