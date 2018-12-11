import numpy as np

def initialize(shape, mode="linear", scale=1.0):        
    if mode == "linear":
        return _linear_init(shape)
    elif mode == "zero":
        return _zero_init(shape)
    elif mode == "normal":
        return _normal_init(shape, scale)
    elif mode == "xavier":
        return _xavier_init(shape)
    elif mode == "he":
        return _he_init(shape)
    else:
        raise NotImplementedError


def _linear_init(shape):
    init = np.linspace(-0.1, 0.1, num=np.prod(shape))
    return init.reshape(shape)


def _zero_init(shape):
    init = np.zeros(shape)
    return init


def _normal_init(shape, scale=1.0):
    init = np.random.randn(*shape) * scale
    return init


def _xavier_init(shape):
    normal = _normal_init(shape)
    return normal / np.sqrt(shape[0])


def _he_init(shape):
    normal = _normal_init(shape)
    return normal / np.sqrt(shape[0] / 2)
