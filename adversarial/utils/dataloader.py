import numpy as np
import tensorflow as tf


def load_data(path):
    X_train = np.load(f'{path}/X_train.npy')
    y_train = np.load(f'{path}/y_train.npy')
    X_test = np.load(f'{path}/X_test.npy')
    y_test = np.load(f'{path}/y_test.npy')

    return X_train, y_train, X_test, y_test


def concat_data(x_batch, y_batch, adv_samples, random_noise=True):
    x_batch_numpy = x_batch.numpy()
    
    if random_noise is True:
        for i in range(x_batch_numpy.shape[0]):
            if np.random.rand() < 0.5:
                img = x_batch_numpy[i]
                img += np.random.normal(loc=0.0, scale=0.1, size=(img.shape))
                img[img < 0] = 0
                img[img > 1] = 1
                x_batch_numpy[i] = img
                
    x_batch = tf.convert_to_tensor(x_batch_numpy, dtype=tf.float32)
    
    x_batch = tf.concat([x_batch, adv_samples], axis=0)
    y_batch = tf.tile(y_batch, [2])
    
    return x_batch, y_batch


class DataLoader():

    HEIGHT = 160
    WIDTH = 160
    CHANNEL = 3
    N_CLASSES = 10

    def __init__(self, path, batch_size, training=True):
        self.batch_size = batch_size
        
        if training is True:
            self.X, self.y, _, _ = load_data(path)
        else:
            _, _, self.X, self.y = load_data(path)

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def __getitem__(self, index):
        return (self.X[index], self.y[index], np.random.randint(0, DataLoader.N_CLASSES))

    def __iter__(self):
        rindex = np.arange(0, self.X.shape[0])
        np.random.shuffle(rindex)

        for b in range(len(self)):
            start_index = b*self.batch_size
            end_index = min(self.X.shape[0], (b+1)*self.batch_size)

            x_batch = np.zeros((self.batch_size, DataLoader.HEIGHT, DataLoader.WIDTH, DataLoader.CHANNEL))
            y_batch = np.zeros((self.batch_size,))
            adv_batch = np.zeros((self.batch_size,))

            for i in range(start_index, end_index):
                r = rindex[i]
                x_batch[i - start_index], y_batch[i - start_index], adv_batch[i - start_index] = self[r]

            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.int32)
            adv_batch = tf.convert_to_tensor(adv_batch, dtype=tf.int32)

            yield x_batch, y_batch, adv_batch
