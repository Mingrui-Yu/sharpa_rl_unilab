"""Shared learning-rate rule for PPO and APPO KL feedback."""


def adaptive_learning_rate(lr: float, kl: float, target: float, kl_factor: float, lr_factor: float):
    """Keep negative/NaN KL out of the low-KL branch; retain upstream LR limits."""
    if kl > target * kl_factor:
        return max(1e-5, lr / lr_factor)
    if 0 <= kl < target / kl_factor:
        return min(1e-2, lr * lr_factor)
    return lr
