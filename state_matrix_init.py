import torch as th


def numpy_like_meshgrid(size):
    x = th.arange(size)
    y = th.arange(size)
    return th.meshgrid(x, y, indexing='ij')


def init_state_space(size, name):
    pass


def warped_scaled_legendre(size):
    n, k = numpy_like_meshgrid(size)

    A = th.tril(th.ones((size, size)), -1)
    A = th.where(n == k, (n + 1) / (2 * n + 1), A)
    A *= -th.sqrt(2 * n + 1) * th.sqrt(2 * k + 1)

    B = th.sqrt((2 * n[:,:1] + 1))

    return A, B


def truncated_fourier(size):
    # TODO: Add once correction is release
    pass