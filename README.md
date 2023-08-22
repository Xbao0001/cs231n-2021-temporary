# cs231n-2021 临时仓库


## Overview 
这是我学习深度学习和计算机视觉（Stanford cs231n）的记录(机器学习见[cs229](https://github.com/Xbao0001/cs229-temporary))。

主要包括深度学习的算法、模块、公式推导和代码实现，均为独立完成。详见[课程大纲](http://cs231n.stanford.edu/2021/schedule.html)

- 比如常见Layer的前向和反向过程，例如Linear, Conv, ReLU, BatchNorm, GroupNorm, MaxPooling等等，详见[layers.py](./assignment2/cs231n/layers.py)

- 常见优化器的实现，例如SGD, Momentum, RMSProp, Adam等，详见[optim.py](./assignment2/cs231n/optim.py)

- 常见模型和算法，例如，对比学习，Image Captioning, Gan, LSTM, Transformer等等，详见[文件夹](./assignment3/cs231n/)

以面试常问的BatchNorm为例：
```Python
# BatchNorm
def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.
    
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        sample_mean = np.mean(x, axis=0, keepdims=True)
        sample_var = np.var(x, axis=0, keepdims=True)

        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_hat + beta
        cache = {
                'x': x,
                'x_hat': x_hat,
                'mu': sample_mean,
                'gamma': gamma,
                'sigma': np.sqrt(sample_var),
                'eps': eps,
        }

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == "test":
        out = gamma * x + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    N, _ = dout.shape
    x_hat = cache['x_hat']
    gamma = cache['gamma']
    sigma = cache['sigma']
    eps = cache['eps']

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = gamma / (N * np.sqrt(sigma**2 + eps)) * (N * dout - np.sum(dout, axis=0) - x_hat * np.sum(dout * x_hat, axis=0))
    return dx, dgamma, dbeta
```




Assignment solution of Stanford CS231n(2021).

Course video: [Stanford 2017](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv), [Michigan 2019](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r), [中文字幕](https://www.bilibili.com/video/BV1Dx411n7UE)

Course schedule & slides: [Stanford cs231n](http://cs231n.stanford.edu/schedule.html), [Michigan](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/schedule.html)

Course notes: [Stanford cs231n](https://cs231n.github.io/)

- Assignment1: Done

- Assignment2: Done

- Assignment3: Done

Note: The zip files are the original assignment files downloaded from the [course webside](https://cs231n.github.io/).

If you have any problems, feel free to open an issue, I am willing to discuss with you.   
