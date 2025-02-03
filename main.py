import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from trainer import (
    train_model,
    load_and_link_data,
    preprocess_image,
    create_id_mapping,  # Ensure this is imported from trainer.py
)


def predict_card(image_path, model_path="pokemon_mobilenet.h5", id_mapping=None, top_n=5):
    """
    Predict the top N Pokémon card IDs using the trained model.
    """
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Training a new model...")
        train_model()

    # Load the trained model
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Reverse mapping to map predicted label to card ID
    reverse_mapping = {v: k for k, v in id_mapping.items()}

    # Preprocess the image
    image_data = preprocess_image(image_path)  # Ensure this outputs (1, 224, 224, 3)

    # Make prediction
    predictions = model.predict(image_data)

    # Get top N predictions
    top_n_indices = np.argsort(predictions[0])[-top_n:][::-1]  # Indices of the top N probabilities
    top_n_probabilities = predictions[0][top_n_indices]  # Probabilities of the top N
    top_n_labels = [reverse_mapping.get(idx, "Unknown") for idx in top_n_indices]  # Get card IDs

    # Display the top N predictions
    print(f"Top {top_n} Predictions:")
    for i in range(top_n):
        print(f"{i+1}. Card ID: {top_n_labels[i]} with probability: {top_n_probabilities[i]:.4f}")


if __name__ == "__main__":
    # Load the linked data and create the ID mapping
    df = load_and_link_data()
    id_mapping = create_id_mapping(df)

    # Path to the test image
    test_image = r"Maushold.jpg"

    # Check if the test image exists and make a prediction
    if os.path.exists(test_image):
        predict_card(test_image, id_mapping=id_mapping)
    else:
        print(f"Test image not found: {test_image}")
