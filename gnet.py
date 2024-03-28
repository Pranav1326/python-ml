import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the GoogLeNet (Inception V1) architecture
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool):
    path1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
    path2 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    path2 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(path2)
    path3 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    path3 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(path3)
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = Conv2D(filters_pool, (1, 1), padding='same', activation='relu')(path4)
    return tf.keras.layers.concatenate([path1, path2, path3, path4], axis=-1)

# Create the GoogLeNet model
input_layer = Input(shape=(224, 224, 3))
x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2')(input_layer)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv_2_3x3/1')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

# Apply multiple inception modules
x = inception_module(x, 64, 96, 128, 16, 32, 32)
x = inception_module(x, 128, 128, 192, 32, 96, 64)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = inception_module(x, 192, 96, 208, 16, 48, 64)
x = inception_module(x, 160, 112, 224, 24, 64, 64)
x = inception_module(x, 128, 128, 256, 24, 64, 64)
x = inception_module(x, 112, 144, 288, 32, 64, 64)
x = inception_module(x, 256, 160, 320, 32, 128, 128)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = inception_module(x, 256, 160, 320, 32, 128, 128)
x = inception_module(x, 384, 192, 384, 48, 128, 128)
x = AveragePooling2D((7, 7), strides=(1, 1))(x)
x = Flatten()(x)
output = Dense(4, activation='softmax')(x)  # Assuming you have 4 classes

# Create the final GoogLeNet model
model = tf.keras.Model(inputs=input_layer, outputs=output)

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
