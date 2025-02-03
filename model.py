import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy('mixed_float16')

def create_mobilenet_model(num_classes=18171):
    """Create a fine-tuned MobileNetV2 model with batch normalization and L2 regularisation."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #base_model.trainable = False  # Freeze the base model to use pre-trained weights
    
    num_layers = len(base_model.layers)
    num_unfreeze = int(num_layers * 0.3)  

    for layer in base_model.layers[:num_layers - num_unfreeze]:
        layer.trainable = False
    for layer in base_model.layers[num_layers - num_unfreeze:]:
        layer.trainable = True

    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.3),
        Dense(512, activation='relu', kernel_regularizer=l2(0.02)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    return model
