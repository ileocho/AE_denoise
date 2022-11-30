import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def preprocess(array):
    # Normalize the array and reshape into the right shape

    array = array.astype(np.float32) / 255.0
    array = array.reshape(len(array), 28, 28, 1)

    return array


def noise(array):
    # Add random gaussian noise to the array

    noise_factor = 0.4
    noise_array = array + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=array.shape)

    return np.clip(noise_array, 0., 1.)


def display(array1, array2):
    # Display 5 random images for each of the two arrays

    n = 5

    indices = np.random.randint(0, len(array1), n)
    images1 = array1[indices, :, :, :]
    images2 = array2[indices, :, :, :]

    plt.figure(figsize=(20, 2))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1) # 1 row, n columns, i-th subplot
        plt.imshow(image1.reshape(28, 28)) # reshape to 28x28
        plt.gray() # grayscale
        ax.get_xaxis().set_visible(False) # Remove the x-axis
        ax.get_yaxis().set_visible(False) # Remove the y-axis

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

train_data = preprocess(x_train)
test_data = preprocess(x_test)

train_data_noisy = noise(train_data)
test_data_noisy = noise(test_data)

display(train_data, train_data_noisy)

# Define the autoencoder model

input = tf.keras.layers.Input(shape=(28, 28, 1))

# Encoder part
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder part
x = tf.keras.layers.Conv2DTranspose(32,  (3, 3), strides=2, activation='relu', padding='same')(x)
x = tf.keras.layers.Conv2DTranspose(32,  (3, 3), strides=2, activation='relu', padding='same')(x)
x = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.models.Model(input, x)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

# train model to recontruct the input image
# autoencoder.fit(x=train_data, y=train_data, epochs=50, batch_size=512, shuffle=True, validation_data=(test_data, test_data))

# predictions = autoencoder.predict(test_data)
# display(test_data, predictions)

# train model to denoise the input image
autoencoder.fit(x=train_data_noisy, y=train_data, epochs=50, batch_size=512, shuffle=True, validation_data=(test_data_noisy, test_data))
predictions = autoencoder.predict(test_data_noisy)
display(test_data_noisy, predictions)