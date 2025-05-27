def load_image(image_path):
    """Load an image from the specified path."""
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def preprocess_data(data_directory):
    """Preprocess images from the specified directory."""
    import os
    import numpy as np
    images = []
    labels = []
    
    for label in os.listdir(data_directory):
        label_dir = os.path.join(data_directory, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                img_array = load_image(img_path)
                images.append(img_array)
                labels.append(label)
    
    return np.array(images), np.array(labels)

def augment_image(image):
    """Apply random augmentations to the input image."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    image = image.reshape((1,) + image.shape)  # Reshape for the generator
    return datagen.flow(image)

def save_model(model, model_name):
    """Save the trained model to a file."""
    model.save(model_name)