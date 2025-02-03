import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from model import create_mobilenet_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
import re
import seaborn as sns
import matplotlib.pyplot as plt


images_folder = 'images'
combined_data_folder = 'combined_data'


def reverse_normalization(img_array):
    """Reverses the MobileNetV2 preprocessing normalization."""
    img_array = img_array + 1.0  # Rescale to [0, 2]
    img_array = img_array * 127.5  # Scale to [0, 255]
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)  # Clip the values to valid pixel range
    return img_array


def visualize_generator_samples(data_generator, label_to_card_id, num_samples=5):
    """Fetches a batch from a generator and visualises images, labels, and Card IDs."""
    batch_images, batch_labels = next(data_generator)  # Fetch one batch

    plt.figure(figsize=(15, 15))

    for i in range(min(num_samples, len(batch_images))):
        image = reverse_normalization(batch_images[i])  # Reverse normalization
        label_index = np.argmax(batch_labels[i])  # Get the class label (index)
        card_id = label_to_card_id.get(label_index, "Unknown")  # Get card ID

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image.astype(np.uint8))  # Ensure image is in valid display range
        plt.title(f"Label: {label_index}\nCard ID: {card_id}")
        plt.axis('off')

    plt.show()


def check_generator_output(generator, label_to_card_id, num_batches=5, num_samples=5):
    """Verifies the generator output and displays sample images with labels and Card IDs."""
    for i in range(num_batches):
        batch_images, batch_labels = next(generator)
        unique_classes = np.unique(np.argmax(batch_labels, axis=1))
        print(f"Batch {i+1}: {len(unique_classes)} unique labels -> {unique_classes}")

        # Select a random subset of samples
        indices = np.random.choice(len(batch_images), num_samples, replace=False)
        sample_images = batch_images[indices]
        sample_labels = batch_labels[indices]

        # Display these samples with labels
        visualize_generator_samples(generator, label_to_card_id, num_samples=num_samples)


def load_and_link_data(limit=None):
    """Load Pokémon card data and link image paths."""
    card_info = []
    for file_name in os.listdir(combined_data_folder):
        if file_name.endswith('.csv'):
            set_data = pd.read_csv(os.path.join(combined_data_folder, file_name))
            for _, row in set_data.iterrows():
                # Ensure 'set' and 'name' are strings before creating the image path
                image_path = os.path.join(images_folder, str(row['set']), f"{str(row['name'])}.jpg")
                if os.path.exists(image_path):
                    card_info.append({
                        'id': str(row['id']),
                        'set': str(row['set']),
                        'image_path': image_path
                    })
    df = pd.DataFrame(card_info)
    return df.head(limit) if limit else df



def preprocess_image(image_path, target_size=(224, 224)):
    """Loads and preprocesses an image for MobileNetV2."""
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # Normalize to [-1, 1]
    
    # Add batch dimension to the image (from (224, 224, 3) to (1, 224, 224, 3))
    #img_array = np.expand_dims(img_array, axis=0)
    
    return img_array



def create_id_mapping(df):
    """
    Create a mapping of card IDs to integer labels.
    """
    unique_ids = df['id'].unique()
    return {card_id: idx for idx, card_id in enumerate(unique_ids)}

def image_generator(df, id_mapping, batch_size=32, target_size=(224, 224), num_classes=None, augment=False):
    """Data generator with augmentation for training."""
    if augment:
        data_gen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            channel_shift_range=30.0,
            horizontal_flip=True,
            fill_mode="nearest"
        )
    else:
        data_gen = ImageDataGenerator()

    while True:
        batch_images, batch_labels = [], []
        for _ in range(batch_size):
            row = df.sample().iloc[0]
            image = preprocess_image(row['image_path'], target_size)
            label = id_mapping[row['id']]
            batch_images.append(image)
            batch_labels.append(label)

        batch_images = np.array(batch_images)
        batch_labels = to_categorical(batch_labels, num_classes)

        augmented_images = next(data_gen.flow(batch_images, batch_size=batch_size, shuffle=False))
        yield augmented_images, batch_labels


def plot_set_distribution(df, title="Set Distribution", highlight_sets=None):
    """Plots the number of cards per set, highlighting specific sets."""
    set_counts = df['set'].value_counts().sort_values()
    avg_cards_per_set = set_counts.mean()

    plt.figure(figsize=(14, 6))
    
    # Highlight sets
    colors = ['red' if set_name in highlight_sets else 'blue' for set_name in set_counts.index]
    
    sns.barplot(x=set_counts.index, y=set_counts, palette=colors)
    plt.axhline(avg_cards_per_set, color='black', linestyle='dashed', linewidth=2, label=f"Average ({int(avg_cards_per_set)})")

    plt.xticks(rotation=90)
    plt.xlabel("Pokémon Card Sets")
    plt.ylabel("Number of Cards")
    plt.title(title)
    plt.legend()
    plt.show()


def find_set_statistics(df):
    """Finds the smallest, largest, and middle set based on card count, and calculates the average."""
    set_counts = df['set'].value_counts()
    avg_cards_per_set = int(set_counts.mean())  # Ensure average is correctly calculated
    
    smallest_set = set_counts.idxmin()
    largest_set = set_counts.idxmax()

    # Find the set closest to the average
    middle_set = (set_counts - avg_cards_per_set).abs().idxmin()

    print(f"📉 Smallest Set: {smallest_set} ({set_counts[smallest_set]} cards)")
    print(f"📈 Largest Set: {largest_set} ({set_counts[largest_set]} cards)")
    print(f"⚖️ Middle Set (Closest to Average): {middle_set} ({set_counts[middle_set]} cards)")
    print(f"🔢 Average Cards Per Set: {avg_cards_per_set}")

    return smallest_set, largest_set, middle_set, avg_cards_per_set  # Return all statistics


def oversample_below_middle(df):
    """Oversamples sets that have fewer than the middle set, ensuring the average remains the same."""
    # Access original card counts and calculate statistics
    smallest_set, largest_set, middle_set, avg_cards_per_set = find_set_statistics(df)
    
    print("\n🔍 Checking set distribution BEFORE oversampling...")
    plot_set_distribution(df, "Set Distribution Before Oversampling", highlight_sets=[smallest_set, largest_set, middle_set])
    
    # Save the original counts before oversampling
    original_counts = df['set'].value_counts()
    
    # Oversample the sets with fewer cards than the original middle set
    set_counts = df['set'].value_counts()
    below_middle_sets = original_counts[original_counts < set_counts[middle_set]].index

    # Oversample only those sets
    oversampled_df = df.copy()
    
    for set_name in below_middle_sets:
        set_size = set_counts[set_name]
        # Calculate the number of additional samples needed to match the middle set size
        additional_samples = set_counts[middle_set] - set_size
        if additional_samples > 0:
            # Get the samples from this set
            samples_to_add = df[df['set'] == set_name].sample(n=additional_samples, replace=True, random_state=42)
            # Add those samples to the dataframe
            oversampled_df = pd.concat([oversampled_df, samples_to_add], ignore_index=True)

    # After oversampling, recalculate the set statistics
    smallest_set, largest_set, middle_set, avg_cards_per_set = find_set_statistics(oversampled_df)
    
    print("\n✅ Checking set distribution AFTER oversampling...")
    plot_set_distribution(oversampled_df, "Set Distribution After Oversampling", highlight_sets=[smallest_set, largest_set, middle_set])
    
    # Ensure that after oversampling, the average is maintained and larger sets are unchanged
    oversampled_set_counts = oversampled_df['set'].value_counts()
    oversampled_avg = oversampled_set_counts.mean()
    
    print(f"Original average card count: {avg_cards_per_set}")
    print(f"New average after oversampling: {oversampled_avg}")
    
    return oversampled_df




def train_model():
    """Train the MobileNetV2 model with balanced data."""
    model_path = "pokemon_mobilenet.h5"
    if os.path.exists(model_path):
        print(f"Model found at {model_path}. Skipping training.")
        return

    df = load_and_link_data()
    train_df, val_df = train_test_split(df, test_size=0.3, stratify=df['set'], random_state=42) #check split 0.2 vs 0.3

    train_df = oversample_below_middle(train_df)  # Oversample ONLY the training data

    id_mapping = create_id_mapping(df)
    num_classes = len(id_mapping)

    train_gen = image_generator(train_df, id_mapping, batch_size=32, num_classes=num_classes, augment=True)
    val_gen = image_generator(val_df, id_mapping, batch_size=32, num_classes=num_classes, augment=False)



    # Check generator output
    label_to_card_id = {idx: card_id for card_id, idx in id_mapping.items()}  # Reverse id_mapping for display
    print("Checking training generator output:")
    check_generator_output(train_gen, label_to_card_id, num_batches=2, num_samples=5)


    print("Checking validation generator output:")
    check_generator_output(val_gen, label_to_card_id, num_batches=2, num_samples=5)

    # Create the model
    model = create_mobilenet_model(num_classes)
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=150,
        steps_per_epoch=len(train_df) // 32,
        validation_steps=len(val_df) // 32,
        callbacks=callbacks
    )

    model.save(model_path)
    print(f"Model saved to {model_path}.")




