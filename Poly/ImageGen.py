import numpy as np
"""
distorting images with noise, same noise generation algorithm with (Becker et al.,2019)
"""
def add_img_noise(imgs, first_n_clean, random, corr=0.2, lowlow=0.0, lowup=0.25, uplow=0.75, upup=1.0):
    """
    imgs: Images to add noise to
    first_n_clean: Keep first_n_images without distortion 
    random: np.random.RandomState 
    corr: "correlation (over time) 
    lowlow: lower bound of the interval the lower bound for each sequence is sampled from
    lowup: upper bound of the interval the lower bound for each sequence is sampled from
    uplow: lower bound of the interval the upper bound for each sequence is sampled from
    upup: upper bound of the interval the upper bound for each sequence is sampled from
    
    """

    assert lowlow <= lowup <= uplow <= upup, "Invalid bounds"
    if len(imgs.shape) < 5:
        imgs = np.expand_dims(imgs, -1)
    batch_size, seq_len = imgs.shape[:2]
    factors = np.zeros([batch_size, seq_len])
    factors[:, 0] = random.uniform(low=0.0, high=1.0, size=batch_size)
    for i in range(seq_len - 1):
        factors[:, i + 1] = np.clip(factors[:, i] + random.uniform(low=-corr, high=corr, size=batch_size), a_min=0.0, a_max=1.0)

    t1 = random.uniform(low=lowlow, high=lowup, size=(batch_size, 1))
    t2 = random.uniform(low=uplow, high=upup, size=(batch_size, 1))

    factors = (factors - t1) / (t2 - t1)
    factors = np.clip(factors, a_min=0.0, a_max=1.0)
    factors = np.reshape(factors, list(factors.shape) + [1, 1, 1])
    factors[:, :first_n_clean] = 1.0
    noisy_imgs = []

    for i in range(batch_size):
        if imgs.dtype == np.uint8:
            noise = random.uniform(low=0.0, high=255, size=imgs.shape[1:])
            noisy_imgs.append((factors[i] * imgs[i] + (1 - factors[i]) * noise).astype(np.uint8))
        else:
            noise = random.uniform(low=0.0, high=1.1, size=imgs.shape[1:])
            noisy_imgs.append(factors[i] * imgs[i] + (1 - factors[i]) * noise)

    return np.squeeze(np.concatenate([np.expand_dims(n, 0) for n in noisy_imgs], 0)), factors


def add_img_noise4(imgs, first_n_clean, random, corr=0.2, lowlow=0.0, lowup=0.25, uplow=0.75, upup=1.0):
    """
    imgs: Images to add noise to
    first_n_clean: Keep first_n_images without distortion 
    random: np.random.RandomState 
    corr: "correlation (over time) 
    lowlow: lower bound of the interval the lower bound for each sequence is sampled from
    lowup: upper bound of the interval the lower bound for each sequence is sampled from
    uplow: lower bound of the interval the upper bound for each sequence is sampled from
    upup: upper bound of the interval the upper bound for each sequence is sampled from
    
    """

    half_x = int(imgs.shape[2] / 2)
    half_y = int(imgs.shape[3] / 2)
    assert lowlow <= lowup <= uplow <= upup, "Invalid bounds"
    if len(imgs.shape) < 5:
        imgs = np.expand_dims(imgs, -1)
    batch_size, seq_len = imgs.shape[:2]
    factors = np.zeros([batch_size, seq_len, 4])
    factors[:, 0] = random.uniform(low=0.0, high=1.0, size=(batch_size, 4))
    for i in range(seq_len - 1):
        factors[:, i + 1] = np.clip(factors[:, i] + random.uniform(low=-corr, high=corr, size=(batch_size, 4)), a_min=0.0, a_max=1.0)

    t1 = random.uniform(low=lowlow, high=lowup, size=(batch_size, 1, 4))
    t2 = random.uniform(low=uplow, high=upup, size=(batch_size, 1, 4))

    factors = (factors - t1) / (t2 - t1)
    factors = np.clip(factors, a_min=0.0, a_max=1.0)
    factors = np.reshape(factors, list(factors.shape) + [1, 1, 1])
    factors[:, :first_n_clean] = 1.0
    noisy_imgs = []
    qs = []
    for i in range(batch_size):
        if imgs.dtype == np.uint8:
            qs.append(detect_pendulums(imgs[i], half_x, half_y))
            noise = random.uniform(low=0.0, high=255, size=[4, seq_len, half_x, half_y, imgs.shape[-1]]).astype(np.uint8)
            curr = np.zeros(imgs.shape[1:], dtype=np.uint8)
            curr[:, :half_x, :half_y] = (factors[i, :, 0] * imgs[i, :, :half_x, :half_y] + (1 - factors[i, :, 0]) * noise[0]).astype(np.uint8)
            curr[:, :half_x, half_y:] = (factors[i, :, 1] * imgs[i, :, :half_x, half_y:] + (1 - factors[i, :, 1]) * noise[1]).astype(np.uint8)
            curr[:, half_x:, :half_y] = (factors[i, :, 2] * imgs[i, :, half_x:, :half_y] + (1 - factors[i, :, 2]) * noise[2]).astype(np.uint8)
            curr[:, half_x:, half_y:] = (factors[i, :, 3] * imgs[i, :, half_x:, half_y:] + (1 - factors[i, :, 3]) * noise[3]).astype(np.uint8)
        else:
            noise = random.uniform(low=0.0, high=1.0, size=[4, seq_len, half_x, half_y, imgs.shape[-1]])
            curr = np.zeros(imgs.shape[1:])
            curr[:, :half_x, :half_y] = factors[i, :, 0] * imgs[i, :, :half_x, :half_y] + (1 - factors[i, :, 0]) * noise[0]
            curr[:, :half_x, half_y:] = factors[i, :, 1] * imgs[i, :, :half_x, half_y:] + (1 - factors[i, :, 1]) * noise[1]
            curr[:, half_x:, :half_y] = factors[i, :, 2] * imgs[i, :, half_x:, :half_y] + (1 - factors[i, :, 2]) * noise[2]
            curr[:, half_x:, half_y:] = factors[i, :, 3] * imgs[i, :, half_x:, half_y:] + (1 - factors[i, :, 3]) * noise[3]
        noisy_imgs.append(curr)

    factors_ext = np.concatenate([np.squeeze(factors), np.zeros([factors.shape[0], factors.shape[1], 1])], -1)
    q = np.concatenate([np.expand_dims(q, 0) for q in qs], 0)
    f = np.zeros(q.shape)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for k in range(3):
                f[i, j, k] = factors_ext[i, j, q[i, j, k]]

    return np.squeeze(np.concatenate([np.expand_dims(n ,0) for n in noisy_imgs], 0)), f


def detect_pendulums(imgs, half_x, half_y):
    qs = [imgs[:, :half_x, :half_y], imgs[:, :half_x, half_y:], imgs[:, half_x:, :half_y], imgs[:, half_x:, half_y:]]

    r_cts = np.array([np.count_nonzero(q[:, :, :, 0] > 5, axis=(-1, -2)) for q in qs]).T
    g_cts = np.array([np.count_nonzero(q[:, :, :, 1] > 5, axis=(-1, -2)) for q in qs]).T
    b_cts = np.array([np.count_nonzero(q[:, :, :, 2] > 5, axis=(-1, -2)) for q in qs]).T

    cts = np.concatenate([np.expand_dims(c, 1) for c in [r_cts, g_cts, b_cts]], 1)

    q_max = np.max(cts, -1)
    q = np.argmax(cts, -1)
    q[q_max < 10] = 4
    return q





