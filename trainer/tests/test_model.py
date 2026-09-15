import torch

from train.losses import ClassBalancedCoralLoss, effective_number_weights, ordinal_targets
from train.model import OrdinalImageModel, coral_logits_to_probs


def test_ordered_coral_head_produces_valid_probabilities():
    model = OrdinalImageModel("tiny_cnn", "test", dropout=0.0, pretrained=False)
    logits = model(torch.randn(4, 3, 32, 32))
    survival = torch.sigmoid(logits)
    assert torch.all(survival[:, :-1] >= survival[:, 1:])
    probabilities = coral_logits_to_probs(logits)
    assert probabilities.shape == (4, 5)
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-6)


def test_class_balanced_coral_loss_is_finite():
    labels = torch.tensor([0, 0, 1, 2, 3, 4])
    weights = effective_number_weights(labels.numpy(), beta=0.99)
    loss = ClassBalancedCoralLoss(weights)(torch.randn(6, 4), labels)
    assert torch.isfinite(loss)
    assert ordinal_targets(torch.tensor([0, 4])).tolist() == [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]

