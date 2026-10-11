import pytest
import torch

from train.losses import ClassBalancedCoralLoss, effective_number_weights, ordinal_targets, smoothed_ordinal_targets


def test_zero_smoothing_is_bit_identical_to_hard_targets():
    labels = torch.tensor([0, 1, 2, 3, 4])
    assert torch.equal(smoothed_ordinal_targets(labels, 5, 0.0), ordinal_targets(labels, 5))


def test_smoothing_softens_toward_the_midpoint_and_stays_bounded():
    labels = torch.tensor([0, 4])
    targets = smoothed_ordinal_targets(labels, 5, 0.1)
    assert targets[0].tolist() == pytest.approx([0.05] * 4)
    assert targets[1].tolist() == pytest.approx([0.95] * 4)
    assert torch.all(targets >= 0.0) and torch.all(targets <= 1.0)


def test_smoothed_loss_penalises_a_confident_correct_model_more():
    """Smoothing must raise the loss where the model is already right and sure."""
    labels = torch.tensor([0, 1, 2, 3, 4])
    weights = effective_number_weights(labels.numpy(), beta=0.999)
    correct_confident = torch.tensor(
        [[-8.0, -8.0, -8.0, -8.0], [8.0, -8.0, -8.0, -8.0], [8.0, 8.0, -8.0, -8.0],
         [8.0, 8.0, 8.0, -8.0], [8.0, 8.0, 8.0, 8.0]]
    )
    hard = ClassBalancedCoralLoss(weights)(correct_confident, labels)
    smoothed = ClassBalancedCoralLoss(weights, label_smoothing=0.1)(correct_confident, labels)
    assert torch.isfinite(smoothed)
    assert float(smoothed) > float(hard)


def test_label_smoothing_defaults_to_off():
    labels = torch.tensor([0, 1, 2, 3, 4])
    weights = effective_number_weights(labels.numpy(), beta=0.999)
    logits = torch.randn(5, 4)
    assert float(ClassBalancedCoralLoss(weights)(logits, labels)) == float(
        ClassBalancedCoralLoss(weights, label_smoothing=0.0)(logits, labels)
    )
