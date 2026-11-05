import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from analysis import model_training as training
from model.base_model import LeastSquareModel, LeastSquareModel2


class LeastSquareTrainingTest(unittest.TestCase):
    def fit(self, model, x, y):
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=3, shuffle=True)
        training.fit_least_square_model(model, loader)

    def test_multiple_targets_and_silent_features(self):
        x = torch.tensor([-46.05, -10, 0, 1, 3, 5]).view(-1, 1, 1).expand(-1, 4, 7)
        y = x[:, 0, 0, None] * torch.tensor([[.17, -.2]]) + torch.tensor([[.18, 2.]])
        model = LeastSquareModel(out_features=2)
        self.fit(model, x, y)
        torch.testing.assert_close(model(x), y)

    def test_power_features(self):
        x = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2) / 10
        feature = torch.log(torch.exp(x).mean((1, 2))).unsqueeze(1)
        y = .4 * feature + .3
        model = LeastSquareModel(power=True)
        self.fit(model, x, y)
        torch.testing.assert_close(model(x), y)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable')
    def test_mps_fit_and_prediction(self):
        x = torch.tensor([-46.05, -10, 0, 1, 3, 5]).view(-1, 1, 1).expand(-1, 4, 7).to('mps')
        y = .17 * x[:, 0, 0, None] + .18
        model = LeastSquareModel().to('mps')
        self.fit(model, x, y)
        self.assertEqual(model.linear.weight.device.type, 'mps')
        self.assertEqual(model.linear.weight.dtype, torch.float32)
        torch.testing.assert_close(model(x).cpu(), y.cpu())

    def test_constant_features(self):
        x = torch.full((6, 2, 3), -46.05)
        y = torch.arange(6, dtype=torch.float32).unsqueeze(1)
        model = LeastSquareModel()
        self.fit(model, x, y)
        torch.testing.assert_close(model(x), torch.full_like(y, y.mean()))

    def test_invalid_inputs(self):
        x, y = torch.zeros(4, 2, 3), torch.zeros(4, 1)
        with self.assertRaisesRegex(ValueError, 'log=True'):
            self.fit(LeastSquareModel(log=True), x, y)
        x[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.fit(LeastSquareModel(), x, y)

    def test_bandwise_regression(self):
        generator = torch.Generator().manual_seed(10)
        x = torch.randn(100, 64, 7, generator=generator)
        weights = torch.randn(64, 2, generator=generator)
        for power in (False, True):
            with self.subTest(power=power):
                features = torch.log(torch.exp(x).mean(2)) if power else x.mean(2)
                y = features @ weights + torch.tensor([.3, -.8])
                model = LeastSquareModel2(out_features=2, power=power)
                self.fit(model, x, y)
                torch.testing.assert_close(model.extract_features(x), features)
                torch.testing.assert_close(model.linear.weight, weights.T)
                torch.testing.assert_close(model(x), y)
                restored = LeastSquareModel2(out_features=2, power=power)
                restored.load_state_dict(model.state_dict())
                torch.testing.assert_close(restored(x), y)

    def test_bandwise_constant_features(self):
        x = torch.full((6, 64, 7), -46.05)
        y = torch.arange(6, dtype=torch.float32).unsqueeze(1)
        model = LeastSquareModel2()
        self.fit(model, x, y)
        torch.testing.assert_close(model(x), torch.full_like(y, y.mean()))
        with self.assertRaisesRegex(ValueError, 'Expected'):
            model(torch.zeros(2, 3, 7))

    def test_validation_target_scale(self):
        x, y = torch.zeros(2, 4, 7), torch.zeros(2, 1)
        for log_scale in (False, True):
            with patch.object(training, 'read_logmel', return_value=(x, y)) as read:
                training.load_dataset(['train'], ['valid'], 'LeastSquareModel2',
                                      {'time_sec': 5}, True, log_scale, ['distance_3'])
                self.assertEqual(read.call_count, 2)
                for call in read.call_args_list:
                    self.assertEqual(call.kwargs['log_scale'], log_scale)

    def test_training_dispatch_and_saved_predictions(self):
        x = torch.arange(8, dtype=torch.float32).view(-1, 1, 1).expand(-1, 64, 3)
        y = .2 * x[:, 0, 0, None] + .1
        # Contradictory validation labels must never influence fitted coefficients.
        validation_y = torch.full_like(y, 100)
        for name in ['LeastSquareModel', 'LeastSquareModel2', 'SimpleCNN2']:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as folder:
                with patch.object(training, 'load_dataset', return_value=((x, y), (x, validation_y))), \
                     patch.object(training, 'view_loss'), \
                     patch.object(training, 'plot_result'), \
                     patch.object(training, 'write_result') as write, \
                     patch.object(training, 'model_train', return_value=1.) as train, \
                     patch.object(training, 'load_model_setting') as settings:
                    settings.return_value = (LeastSquareModel(), None, torch.nn.MSELoss())
                    training.audio_crowd_training([], [], folder, name, {}, ['distance_3'],
                                                  epoch=2, dev='cpu', batch_size=3, log_scale=True)
                    state = torch.load(Path(folder)/f'{name}_model.pt', weights_only=True)
                    self.assertEqual(write.call_count, 2)
                    if name in ('LeastSquareModel', 'LeastSquareModel2'):
                        train.assert_not_called()
                        settings.assert_not_called()
                        expected = torch.full((1, 64), .2 / 64) if name == 'LeastSquareModel2' else torch.tensor([[.2]])
                        torch.testing.assert_close(state['linear.weight'], expected)
                        torch.testing.assert_close(state['linear.bias'], torch.tensor([.1]))
                        prediction = torch.from_numpy(write.call_args.args[2])
                        torch.testing.assert_close(prediction, y)
                        self.assertTrue(write.call_args.kwargs['log_scale'])
                    else:
                        self.assertEqual(train.call_count, 2)
                        settings.assert_called_once()


if __name__ == '__main__':
    unittest.main()
