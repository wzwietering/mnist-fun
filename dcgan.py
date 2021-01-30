from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, Flatten, LeakyReLU, Conv2D, UpSampling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

class DCGan():
    def __init__(self):
        pass

    def get_generator(self):
        generator = Sequential()
        generator.add(Dense(128*7*7, input_dim=100,
                      kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Reshape((7, 7, 128)))
        generator.add(UpSampling2D(size=(2, 2)))

        generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
        generator.add(LeakyReLU(0.2))
        generator.add(UpSampling2D(size=(2, 2)))
        
        generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
        return generator

    def get_discriminator(self):
        discriminator = Sequential()
        discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        
        discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Flatten())
        
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return discriminator

    def get_gan(self, generator=None, discriminator=None):
        if not generator: generator = self.get_generator()
        if not discriminator: discriminator = self.get_discriminator()
        discriminator.trainable = False
        gan_input = Input(shape=(100,))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return gan
