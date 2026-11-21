import io
import unittest
from unittest.mock import patch

import torch

from easyocr.detection import get_detector


class CraftCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.layer = torch.nn.BatchNorm2d(2)
        self.layer.running_mean.fill_(3)

    def load(self, state):
        checkpoint = io.BytesIO()
        torch.save(state, checkpoint)
        checkpoint.seek(0)
        devices = []

        def make_layer():
            layer = torch.nn.BatchNorm2d(2)
            devices.append(layer.weight.device.type)
            return layer

        with patch('easyocr.detection.CRAFT', side_effect=make_layer):
            net = get_detector(checkpoint, device='cpu', quantize=False)
        return net, devices

    @unittest.skipUnless(hasattr(torch.device, '__enter__'), 'requires a device context')
    def test_complete_checkpoint_constructs_on_meta_and_loads_on_cpu(self):
        state = self.layer.double().state_dict()
        net, devices = self.load({'module.' + key: value for key, value in state.items()})
        self.assertEqual(devices, ['meta'])
        self.assertEqual(net.weight.dtype, torch.float32)
        self.assertEqual(net.weight.device.type, 'cpu')
        torch.testing.assert_close(net.running_mean, self.layer.running_mean.float())

    def test_old_checkpoint_gets_initialized_batch_counter(self):
        state = self.layer.state_dict()
        del state['num_batches_tracked']
        net, _ = self.load(state)
        self.assertEqual(net.num_batches_tracked.item(), 0)
        torch.testing.assert_close(net.running_mean, self.layer.running_mean)

    def test_missing_parameter_still_fails_strict_loading(self):
        state = self.layer.state_dict()
        del state['weight']
        with self.assertRaisesRegex(RuntimeError, 'weight'):
            self.load(state)
