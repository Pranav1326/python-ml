import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained VGG16 model (without top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
class_names = open("./labels.txt").readlines()
output = Dense(len(class_names), activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create an ImageDataGenerator object
datagen = ImageDataGenerator(validation_split=0.2)  # assuming 20% of the data is for validation

# Load images from the directory
train_generator = datagen.flow_from_directory(
    'images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training')  # set as training data

validation_generator = datagen.flow_from_directory(
    'images',  # same directory as training data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')  # set as validation data

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)