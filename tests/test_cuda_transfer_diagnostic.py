"""Offline tests for the model-free CUDA copy diagnostic."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, call, patch


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("diagnose_cuda_transfer", ROOT / "src/diagnose_cuda_transfer.py")
diagnostic = module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class CudaTransferDiagnosticTests(unittest.TestCase):
    def test_comparison_distinguishes_zeros_and_nonfinite_values(self):
        self.assertEqual(diagnostic.compare_values([-3, 5], [-3, 5]),
                         {"exactly_equal": True, "max_abs_difference": 0})
        self.assertEqual(diagnostic.compare_values([-3, 5], [0, 0]),
                         {"exactly_equal": False, "max_abs_difference": 5})
        self.assertEqual(diagnostic.compare_values([-3, 5], [float("inf"), 5]),
                         {"exactly_equal": False, "max_abs_difference": None})

    def test_synchronized_readback_detects_peer_zeros_and_source_mutation(self):
        torch = SimpleNamespace(cuda=Mock())
        source = Mock(device="cuda:0", shape=(2,), dtype="bfloat16")
        destination = Mock(device="cuda:1")
        source.to.return_value = destination
        source.detach.return_value.cpu.return_value.flatten.return_value.tolist.side_effect = [[-3, 5], [-3, 4]]
        destination.detach.return_value.cpu.return_value.flatten.return_value.tolist.return_value = [0, 0]
        timeline = Mock()
        timeline.attach_mock(torch.cuda.synchronize, "sync")
        timeline.attach_mock(source.detach, "source_snapshot")
        timeline.attach_mock(source.to, "transfer")
        timeline.attach_mock(destination.detach, "destination_snapshot")
        _, result = diagnostic.check_transfer(torch, source, "cuda:1", [-3, 5])
        self.assertEqual([entry for entry in timeline.mock_calls if "." not in entry[0]], [
            call.sync(0), call.sync(1), call.source_snapshot(), call.transfer("cuda:1"),
            call.sync(0), call.sync(1), call.destination_snapshot(), call.source_snapshot(),
        ])
        self.assertTrue(result["source_vs_expected"]["exactly_equal"])
        self.assertFalse(result["source_after_vs_before"]["exactly_equal"])
        self.assertEqual(result["source_after_vs_before"]["max_abs_difference"], 1)
        self.assertEqual(result["destination_zero_count"], 2)
        self.assertEqual(result["destination_vs_source_before"]["max_abs_difference"], 5)
        self.assertFalse(result["passed"])

    def test_matrix_covers_both_dtypes_shapes_and_copy_directions(self):
        torch = SimpleNamespace(cuda=Mock(), version=SimpleNamespace(cuda="test CUDA"),
                                __version__="test torch", float32="fp32", bfloat16="bf16",
                                int32="int32", arange=MagicMock())
        torch.cuda.device_count.return_value = 2
        torch.cuda.get_device_name.side_effect = ["GPU zero", "GPU one"]
        torch.cuda.can_device_access_peer.side_effect = [True, False]
        cpu = Mock(device="cpu")
        indices = torch.arange.return_value
        signed = (indices % 126) * 2 - 125
        signed.reshape.return_value.to.return_value = cpu

        def fake_copy(_torch, source, target, _expected):
            destination = Mock(device=target)
            return destination, {
                "source": source.device, "destination": target,
                "passed": (source.device, target) != ("cuda:1", "cuda:0"),
            }

        with patch.object(diagnostic, "cpu_snapshot", return_value=[-125, -123]), \
                patch.object(diagnostic, "check_transfer", side_effect=fake_copy), \
                patch("builtins.print") as log:
            self.assertEqual(diagnostic.run_diagnostic(torch), 1)
        rows = [json.loads(entry.args[0]) for entry in log.call_args_list]
        self.assertEqual(rows[0]["device_names"], ["GPU zero", "GPU one"])
        self.assertEqual(rows[0]["peer_access"], {"cuda:0_to_cuda:1": True, "cuda:1_to_cuda:0": False})
        self.assertEqual([(row["source"], row["destination"]) for row in rows[1:-1]], [
            ("cpu", "cuda:0"), ("cpu", "cuda:1"), ("cuda:0", "cuda:1"), ("cuda:1", "cuda:0"),
        ] * 4)
        self.assertEqual(rows[-1], {"event": "cuda_transfer_summary", "passed": False})
        self.assertEqual(signed.reshape.call_args_list, [call((16,))] * 2 + [call((1, 100, 8192))] * 2)
        self.assertEqual(signed.reshape.return_value.to.call_args_list,
                         [call(dtype="fp32"), call(dtype="bf16")] * 2)


if __name__ == "__main__":
    unittest.main()
