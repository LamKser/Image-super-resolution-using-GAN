from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, Flatten, Dense
from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow.keras.layers import add
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
import tensorflow as tf


class SRGAN:
    def __init__(self, num_res_block=16):
        self.num_res_block = num_res_block

    def residual_block(self, ip):
        residual_model = Conv2D(64, (3, 3), padding="same")(ip)
        residual_model = BatchNormalization(momentum=0.5)(residual_model)
        residual_model = PReLU(shared_axes=[1, 2])(residual_model)

        residual_model = Conv2D(64, (3, 3), padding="same")(residual_model)
        residual_model = BatchNormalization(momentum=0.5)(residual_model)

        return add([ip, residual_model])

    def upscale_block(self, ip):
        upscale_model = Conv2D(256, (3, 3), padding="same")(ip)
        upscale_model = UpSampling2D(size=2)(upscale_model)
        # upscale_model = tf.nn.depth_to_space(upscale_model, 2)
        upscale_model = PReLU(shared_axes=[1, 2])(upscale_model)

        return upscale_model

    def generator(self, gen_ip):
        # rescale = tf.keras.Rescaling(scale=(1.0 / 255.0))(gen_ip)
        layers = Conv2D(64, (9, 9), padding="same")(gen_ip)
        layers = PReLU(shared_axes=[1, 2])(layers)

        temp = layers

        for i in range(self.num_res_block):
            layers = self.residual_block(layers)

        layers = Conv2D(64, (3, 3), padding="same")(layers)
        layers = BatchNormalization(momentum=0.5)(layers)
        layers = add([layers, temp])

        layers = self.upscale_block(layers)
        layers = self.upscale_block(layers)

        op = Conv2D(3, (9, 9), padding="same", activation='tanh')(layers)

        return Model(inputs=gen_ip, outputs=op, name="Generator")

    def discriminator_block(self, ip, filters, strides=1, bn=True):
        disc_model = Conv2D(filters, (3, 3), strides=strides, padding="same")(ip)

        if bn:
            disc_model = BatchNormalization(momentum=0.8)(disc_model)

        disc_model = LeakyReLU(alpha=0.2)(disc_model)

        return disc_model

    def discriminator(self, disc_ip):
        df = 64

        d1 = self.discriminator_block(disc_ip, df, bn=False)
        d2 = self.discriminator_block(d1, df, strides=2)
        d3 = self.discriminator_block(d2, df * 2)
        d4 = self.discriminator_block(d3, df * 2, strides=2)
        d5 = self.discriminator_block(d4, df * 4)
        d6 = self.discriminator_block(d5, df * 4, strides=2)
        d7 = self.discriminator_block(d6, df * 8)
        d8 = self.discriminator_block(d7, df * 8, strides=2)

        d8_5 = Flatten()(d8)
        d9 = Dense(df * 16)(d8_5)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(disc_ip, validity, name="Discriminator")

    def build_vgg(self, ip):
        vgg = VGG19(weights="imagenet", include_top=False, input_shape=ip)
        return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output, name="VGG19")

    def gan(self, gen_model, disc_model, vgg, lr_ip, hr_ip):
        gen_img = gen_model(lr_ip)

        gen_features = vgg(gen_img)

        disc_model.trainable = False
        validity = disc_model(gen_img)

        return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features], name="SRGAN")
