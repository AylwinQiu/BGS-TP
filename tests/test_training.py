"""Small CPU checks; no dataset download or GPU required."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data import load_task_data, split_batch
from src.revnet import COUPLING_MAX_NORM, RevNet, coupling_activation
from src.training import Trainer, evaluate


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_inverse_recovers_forward_states(self):
        for depth in (2, 5, 6):
            model = RevNet(depth, 8).double()
            x1, x2 = torch.randn(2, 3, 8, dtype=torch.float64)
            with torch.no_grad():
                states = model.states(x1, x2)
                targets = model.targets(*states[-1])
                for actual, target in zip(states, targets):
                    for a, b in zip(actual, target):
                        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
                recovered = targets[0][0] - coupling_activation(model.layers[0](x2))
                torch.testing.assert_close(recovered, x1, rtol=1e-12, atol=1e-12)

    def test_norm_clip_preserves_small_deltas_and_direction(self):
        x = torch.randn(4, 8)
        small = torch.nn.functional.leaky_relu(x)
        torch.testing.assert_close(coupling_activation(x), small)
        large = coupling_activation(x * 1000)
        torch.testing.assert_close(large.norm(dim=1), torch.full((4,), COUPLING_MAX_NORM))
        torch.testing.assert_close(large / large.norm(dim=1, keepdim=True),
                                   small / small.norm(dim=1, keepdim=True))

    def test_single_update_changes_only_selected_layer(self):
        model = RevNet(6, 8)
        before = copy.deepcopy(model)
        Trainer(model, layer_select="round-robin").step(torch.rand(3, 16))
        changed = [any(not torch.equal(a, b) for a, b in zip(old.parameters(), new.parameters()))
                   for old, new in zip(before.layers, model.layers)]
        self.assertEqual(changed, [True, False, False, False, False, False])

    def test_block_sweep_matches_fresh_sequential_updates(self):
        block = RevNet(5, 8)
        sequential = copy.deepcopy(block)
        block_trainer = Trainer(block, layer_select="round-robin", layers_per_step=3)
        one_trainer = Trainer(sequential, layer_select="round-robin")
        # Two blocks also exercise the round-robin wrap across depth.
        for _ in range(2):
            images = torch.rand(3, 16)
            block_trainer.step(images)
            for _ in range(3):
                one_trainer.step(images)
        for a, b in zip(block.parameters(), sequential.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_batched_evaluation_weights_last_batch(self):
        model, images = RevNet(4, 8), torch.rand(7, 16)
        x1, x2, truth = split_batch(images)
        with torch.no_grad():
            expected = torch.nn.functional.mse_loss(model(x1, x2)[1], truth).item()
        self.assertAlmostEqual(evaluate(model, images, batch_size=3), expected, places=6)

    def test_all_methods_fit_one_sample(self):
        images = torch.rand(1, 16)
        for method in ("bp", "jacobi", "bgstp"):
            with self.subTest(method=method):
                torch.manual_seed(0)
                model = RevNet(6, 8)
                trainer = Trainer(model, method, lr=0.01)
                initial = evaluate(model, images)
                for step in range(1, 251):
                    trainer.step(images, step, 250)
                self.assertLess(evaluate(model, images), initial * 0.05)

    def test_cifar_layout_and_mask(self):
        images = np.zeros((3, 32, 32, 3), dtype=np.uint8)
        images[:, :, :, 0], images[:, :, :, 1], images[:, :, :, 2] = 255, 128, 64
        with patch("src.data.CIFAR10") as dataset:
            dataset.return_value.data = images
            train, test = load_task_data("unused", train_samples=2, test_samples=1)
        self.assertEqual(train.shape, (2, 3072))
        self.assertEqual(test.shape, (1, 3072))
        torch.testing.assert_close(train[:, :1024], torch.ones(2, 1024))
        torch.testing.assert_close(train[:, 1024:2048], torch.full((2, 1024), 128 / 255))
        torch.testing.assert_close(train[:, 2048:], torch.full((2, 1024), 64 / 255))
        context, mask, truth = split_batch(train)
        self.assertEqual(context.shape, (2, 1536))
        self.assertEqual(torch.count_nonzero(mask).item(), 0)
        torch.testing.assert_close(truth, train[:, 1536:])


if __name__ == "__main__":
    unittest.main()
