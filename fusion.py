import numpy as np

def feature_weight(F):

    eps = 1e-8
    F = (F - F.min()) / (F.max() - F.min() + eps)

    th = np.percentile(F, 90)
    target = F[F >= th]
    background = F[F < th]

    if len(target) == 0 or len(background) == 0:
        return 0.1

    contrast = target.mean() - background.mean()
    noise = background.std() + eps
    peak = F.max()
    sparsity = np.mean(F > 0.7)

    return (contrast / noise) * peak * (1 - sparsity)


def fusion(S_rgb, S_pol, E):

    eps = 1e-8

    S_rgb /= (S_rgb.max() + eps)
    S_pol /= (S_pol.max() + eps)
    E     /= (E.max() + eps)

    w1 = feature_weight(S_rgb)
    w2 = feature_weight(S_pol)
    w3 = feature_weight(E) * 0.3

    if w2 > 1.5 * w1:
        w2 *= 1.5
        w1 *= 0.6
    elif w1 > 1.5 * w2:
        w1 *= 1.5
        w2 *= 0.6

    s = w1 + w2 + w3 + eps

    alpha = w1 / s
    beta  = w2 / s
    gamma = w3 / s

    print(f"alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")

    return alpha*S_rgb + beta*S_pol + gamma*E