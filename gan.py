from keras import initializers
from keras.layers import Dense, Dropout, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam

class Gan():
    def __init__(self):
        pass

    def get_generator(self):
        generator = Sequential()
        generator.add(Dense(256, input_dim=100, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(784, activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return generator

    def get_discriminator(self):
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=784, 
                          kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

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
