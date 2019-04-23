import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.datasets import mnist
from tqdm import tqdm

import data
import gan
import dcgan

# Source https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3

def load_data():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = (trainX.astype(np.float32) - 127.5)/127.5
    trainX = trainX.reshape(60000, 784)
    return (trainX, trainY, testX, testY)

def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation="nearest")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"images/gan_generated_image_{epoch}.png")
    plt.close()

def train(epochs=1, batch_size=128, dc=True):
    if not os.path.exists("images"): os.mkdir("images")
    if not os.path.exists("models"): os.mkdir("models")

    if dc:
        GAN = dcgan.DCGan()
        trainX, trainY, testX, testY = data.get_data(True)
    else:
        GAN = gan.Gan()
        trainX, trainY, testX, testY = load_data()#data.get_data(flatten=True)

    generator = GAN.get_generator()
    discriminator = GAN.get_discriminator()
    ga_network = GAN.get_gan(generator, discriminator)

    for e in range(1, epochs + 1):
        print(f"Epoch {e}")
        for _ in tqdm(range(batch_size)):
            # Generate random noise as an input to initialize the  generator
            noise = np.random.normal(0, 1, [batch_size, 100])

            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)

            # Get a random set of  real images
            image_batch = trainX[np.random.randint(low=0, high=trainX.shape[0], size=batch_size)]

            # Construct different batches of real and fake data 
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9

            #Pre train discriminator on  fake and real data  before starting the ga_network. 
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            #Tricking the noised input of the Generator as real data
            noise = np.random.normal(0, 1, [batch_size, 100])
            y_gen = np.ones(batch_size)

            # During the training of ga_network, 
            # the weights of discriminator should be fixed. 
            #We can enforce that by setting the trainable flag
            discriminator.trainable = False

            #training  the GAN by alternating the training of the Discriminator 
            #and training the chained GAN model with Discriminator’s weights freezed.
            ga_network.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:
            plot_generated_images(e, generator)
            generator.save(f"models/gan_generator_epoch_{e}.h5")
            discriminator.save(f"models/gan_discriminator_epoch_{e}.h5")

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    train(400, dc=False)
