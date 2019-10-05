import numpy as np

##########################################################
#                   Training Utils                       #
##########################################################

def minibatch(X, batchsize = 256, shuffle=True):
    """
    Compute a minibatch indices for a training dataset

    :param X:
        The dataset to divide into minibatches. Assumes the first dimension
        represents the number of training examples.
    :param batchsize: int
        The desired size of each minibatch. Note, however, that if ``X.shape[0] %
        batchsize > 0`` then the final batch will contain fewer than batchsize
        entries. Default is 256.
    :param shuffle:
        Whether to shuffle the entries in the dataset before dividing into
        minibatches. Default is True.
    :return:
        mb_generator : generator
            A generator which yields the indices into X for each batch
        n_batches: int
            The number of batches
    """
    N = X.shape[0]
    idx = np.arange(N)
    nbatches = int(np.ceil(N / batchsize))

    if shuffle:
        np.random.shuffle(idx)

    def mb_generator():
        for i in range(nbatches):
            yield idx[i * batchsize, (i+1) * batchsize]

    return mb_generator(), nbatches


##########################################################
#                   Padding Utils                       #
##########################################################

