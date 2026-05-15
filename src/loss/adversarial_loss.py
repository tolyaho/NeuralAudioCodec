import torch
import torch.nn.functional as F


def discriminator_hinge_loss(
    real_logits: torch.Tensor | list[torch.Tensor],
    fake_logits: torch.Tensor | list[torch.Tensor],
) -> torch.Tensor:
    if isinstance(real_logits, torch.Tensor):
        real_logits = [real_logits]
        fake_logits = [fake_logits]

    loss = sum(
        F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
        for r, f in zip(real_logits, fake_logits)
    )
    return loss / len(real_logits)


def generator_hinge_loss(fake_logits: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(fake_logits, torch.Tensor):
        fake_logits = [fake_logits]
    return sum(-f.mean() for f in fake_logits) / len(fake_logits)


def feature_matching_loss(real_features: list, fake_features: list) -> torch.Tensor:
    if real_features and isinstance(real_features[0], list):
        real_features = [f for scale in real_features for f in scale]
        fake_features = [f for scale in fake_features for f in scale]

    if len(real_features) != len(fake_features):
        raise ValueError(
            f"Feature count mismatch: {len(real_features)} real, "
            f"{len(fake_features)} fake"
        )

    loss = 0.0
    for real, fake in zip(real_features, fake_features):
        loss = loss + F.l1_loss(fake, real.detach())

    return loss / len(real_features)
