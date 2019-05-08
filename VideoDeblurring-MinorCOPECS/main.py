import glob as gb # yeh kyun

import numpy as np
from PIL import Image

import data_utils
from losses import adversarial_loss, generator_loss
from model import generator_model, discriminator_model, generator_containing_discriminator


def train(batch_size, epoch_num):
    # Note the x(blur) in the second, the y(full) in the first
    y_train, x_train = data_utils.load_data(data_type='train')

    # GAN
    g = generator_model()
    d = discriminator_model()
    d_on_g = generator_containing_discriminator(g, d)

    # compile the models, use default optimizer parameters
    # generator use adversarial loss


    g.load_weights('C:/Users/tanay/Downloads/VideoDeblurring-MinorCOPECS/weight/generator_weights.h5')
    g.compile(optimizer='adam', loss=generator_loss)
    # discriminator use binary cross entropy loss
    d.load_weights('C:/Users/tanay/Downloads/VideoDeblurring-MinorCOPECS/weight/discriminator_weights.h5')
    d.compile(optimizer='adam', loss='binary_crossentropy')
    

    # adversarial net use adversarial loss
    d_on_g.compile(optimizer='adam', loss=adversarial_loss)

    for epoch in range(epoch_num):
        print('epoch: ', epoch + 1, '/', epoch_num)
        print('batches: ', int(x_train.shape[0] / batch_size))
        #YEH KYA HAI UPAR
        for index in range(int(x_train.shape[0] / batch_size)):
            # select a batch data
            image_blur_batch = x_train[index * batch_size:(index + 1) * batch_size]
            image_full_batch = y_train[index * batch_size:(index + 1) * batch_size]
            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)
            # output generated images for each 30 iters
          

            # concatenate the full and generated images,
            # the full images at top, the generated images at bottom
            x = np.concatenate((image_full_batch, generated_images))

            # generate labels for the full and generated images
            y = [1] * batch_size + [0] * batch_size

            # train discriminator
            d_loss = d.train_on_batch(x, y)
            print('batch %d d_loss : %f' % (index + 1, d_loss))

            # let discriminator can't be trained
            d.trainable = False

            # train adversarial net
            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [1] * batch_size)
            print('batch %d d_on_g_loss : %f' % (index + 1, d_on_g_loss))

            # train generator
            g_loss = g.train_on_batch(image_blur_batch, image_full_batch)
            print('batch %d g_loss : %f' % (index + 1, g_loss))

            # let discriminator can be trained
            d.trainable = True

            # output weights for generator and discriminator each 30 iters
            # if (index % 30 == 0) and (index != 0):
    g.save_weights('C:/Users/tanay/Downloads/VideoDeblurring-MinorCOPECS/weight/generator_weights.h5', True)
    d.save_weights('C:/Users/tanay/Downloads/VideoDeblurring-MinorCOPECS/weight/discriminator_weights.h5', True)
                # g.save('C:/Users/tanay/Downloads/ImageDeblurring-master/ImageDeblurring-master/weight/generator_model.h5', True)
                # d.save('C:/Users/tanay/Downloads/ImageDeblurring-master/ImageDeblurring-master/weight/discriminator_model.h5', True)


def test(batch_size):
    # Note the x(blur) in the second, the y(full) in the first
    y_test, x_test = data_utils.load_data(data_type='test')
    g = generator_model()
    g.load_weights('C:/Users/tanay/Downloads/VideoDeblurring-MinorCOPECS/weight/generator_weights.h5')
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    data_utils.generate_image(y_test, x_test, generated_images, 'C:/Users/tanay/Downloads/VideoDeblurring-MinorCOPECS/result/')


def test_pictures(batch_size):
    data_path = 'data/test/*.jpeg'
    images_path = gb.glob(data_path)
    data_blur = []
    for image_path in images_path:
        image_blur = Image.open(image_path)
        data_blur.append(np.array(image_blur))

    data_blur = np.array(data_blur).astype(np.float32)
    data_blur = data_utils.normalization(data_blur)

    g = generator_model()
    g.load_weights('weight/generator_weights.h5')
    generated_images = g.predict(x=data_blur, batch_size=batch_size)
    generated = generated_images * 127.5 + 127.5
    for i in range(generated.shape[0]):
        image_generated = generated[i, :, :, :]
        Image.fromarray(image_generated.astype(np.uint8)).save('result/test/' + str(i) + '.png')


if __name__ == '__main__':
    # train(batch_size=4, epoch_num=30)
    test(4)
    # test_pictures(1)
