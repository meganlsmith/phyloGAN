"""CNN discriminator and training functions."""

# python imports
from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    MaxPooling2D, Dropout
from tensorflow.keras import Model


class discriminator(Model):
    """Basic CNN for comparing phylogenetic datasets.
    The shape of the training data is (n, S, 4), where
    n is the number of species, S is the number of sites,
    and 5 is the number of channels (A, T, C, G).
    In addition, we have an extra channel with the proportion of invariant sites.
    We combine convolutional layers for these two classes in the second layer of our CNN.
    Architecture adapted from Wang et al. (2021)"""

    def __init__(self):
        super(discriminator, self).__init__()

        # We use (1, 5) for permutation invariance following Wang et al. (2021)

        self.conv1 = Conv2D(10, (4,1), (4,1), activation='relu')
        self.conv2 = Conv2D(10, (2,1), (4,1), activation='relu')

        self.pool = MaxPooling2D(pool_size = (1,2), strides = (1,2))

        self.flatten = Flatten()
        self.dropout = Dropout(rate=0.2)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')

        self.dense3b = Dense(1, activation = 'linear')#2, activation='softmax') # value is the number of classes






    def call_mathieson(self, x, training=None):
        """x is the genotype matrix, dist is the SNP distances"""

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        x = self.conv1(x)
        x = self.pool(x) # pool
        x = self.conv2(x)
        x = self.pool(x) # pool


        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return self.dense3b(x)

