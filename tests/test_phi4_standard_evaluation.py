"""Offline tests for the standard Phi-4 evaluator."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("evaluate_phi4_standard", ROOT / "src/31_evaluate_phi4_standard.py")
evaluator = module_from_spec(spec)
spec.loader.exec_module(evaluator)


class StandardPhi4EvaluationTests(unittest.TestCase):
    def test_shared_benchmark_and_output_paths(self):
        for dataset, size in (("umwp", 5200), ("sum", 10944)):
            with self.subTest(dataset=dataset):
                rows, digest = evaluator.shared.load_dataset(ROOT, dataset)
                self.assertEqual(len(rows), size)
                self.assertEqual(digest, evaluator.shared.DATASETS[dataset][-1])
                self.assertEqual(evaluator.output_path(ROOT, dataset, "run_01"),
                                 ROOT / f"results/llm/phi-4/{dataset}/run_01")

    def test_config_requires_local_unquantized_phi_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            config = {"model_type": "phi3", "architectures": ["Phi3ForCausalLM"],
                      "max_position_embeddings": 16384}
            config_path = path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            shard = path / "model-00001-of-00006.safetensors"
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"lm_head.weight": shard.name}}), encoding="utf-8",
            )
            with patch.object(evaluator, "MODEL_PATH", path):
                with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                    evaluator.load_config()
                shard.touch()
                self.assertEqual(evaluator.load_config(), config)
                config_path.write_text(json.dumps({**config, "quantization_config": {}}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unquantized"):
                    evaluator.load_config()
                config_path.write_text(json.dumps({**config, "max_position_embeddings": 32768}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "standard Phi-4"):
                    evaluator.load_config()

    def set_up_generator(self):
        torch = SimpleNamespace(bfloat16="bf16", device=lambda name: name,
                                inference_mode=lambda: lambda function: function)
        transformers = SimpleNamespace(AutoTokenizer=Mock(), Phi3ForCausalLM=Mock(),
                                       GenerationConfig=Mock())
        model = Mock()
        model.is_quantized = False
        model.parameters.return_value = [SimpleNamespace(device="cuda:0", dtype="bf16")]
        model.generation_config.eos_token_id = [100257, 100265]
        transformers.Phi3ForCausalLM.from_pretrained.return_value = model, {}
        tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
        tokenizer.chat_template = "native ChatML template"
        tokenizer.eos_token_id = 100265
        tokenizer.pad_token_id = 100349
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>user<|im_sep|>template check<|im_end|><|im_start|>assistant<|im_sep|>"
        )
        return torch, transformers, tokenizer, model

    def test_native_chat_and_strict_decode(self):
        torch, transformers, tokenizer, model = self.set_up_generator()
        inputs = {"input_ids": SimpleNamespace(shape=(1, 10)), "attention_mask": object()}
        batch = Mock()
        batch.to.return_value = inputs
        rendered = tokenizer.apply_chat_template.return_value
        tokenizer.apply_chat_template.side_effect = lambda *_, tokenize, **__: batch if tokenize else rendered
        generated = MagicMock()
        tokens = generated.__getitem__.return_value.tolist
        model.generate.return_value = generated
        generate = evaluator.load_generator(torch, transformers)
        prompt = evaluator.shared.PROMPT.format(question="How many?")
        reasoning = "<think>There is enough information.</think> ANSWERABLE"

        tokens.return_value = [100350, 42, 100351, 9, 100265]
        tokenizer.decode.return_value = reasoning
        self.assertEqual(generate(prompt), (reasoning, tokens.return_value, False))
        self.assertEqual(evaluator.shared.parse_label(reasoning), "INVALID")
        tokenizer.decode.assert_called_once_with(
            [100350, 42, 100351, 9], skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        tokenizer.apply_chat_template.assert_called_with(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        batch.to.assert_called_once_with("cuda:0")
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            evaluator.MODEL_PATH, local_files_only=True,
        )
        transformers.Phi3ForCausalLM.from_pretrained.assert_called_once_with(
            evaluator.MODEL_PATH, local_files_only=True, dtype="bf16", device_map={"": 0},
            use_safetensors=True, output_loading_info=True,
        )
        transformers.GenerationConfig.assert_called_once_with(
            do_sample=False, num_beams=1, max_new_tokens=16, eos_token_id=[100257, 100265],
            pad_token_id=100349, use_cache=True,
        )
        model.generate.assert_called_once_with(**inputs, generation_config=transformers.GenerationConfig.return_value)
        generated.__getitem__.assert_called_once_with((0, slice(10, None)))

        tokens.return_value = [100350] + [42] * 15
        self.assertTrue(generate(prompt)[2])
        tokenizer.decode.assert_called_with(
            tokens.return_value, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        tokens.return_value = [42] * 15 + [100265]
        self.assertFalse(generate(prompt)[2])

    def test_rejects_reasoning_template_before_loading_weights(self):
        torch, transformers, tokenizer, _ = self.set_up_generator()
        tokenizer.apply_chat_template.return_value = "<|im_start|>system<|im_sep|>Think step by step."
        with self.assertRaisesRegex(ValueError, "standard Phi-4 native ChatML"):
            evaluator.load_generator(torch, transformers)
        transformers.Phi3ForCausalLM.from_pretrained.assert_not_called()

    def test_full_run_smoke_gate_and_resume(self):
        torch = SimpleNamespace(cuda=Mock(), version=SimpleNamespace(cuda="test"), __version__="test")
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_properties.return_value = SimpleNamespace(
            name="NVIDIA A100 80GB", total_memory=80 * 1024**3,
        )
        torch.cuda.is_bf16_supported.return_value = True
        torch.cuda.max_memory_allocated.return_value = 0
        torch.cuda.max_memory_reserved.return_value = 0
        transformers = SimpleNamespace(Phi3ForCausalLM=Mock(), __version__="test")
        config = {"model_type": "phi3", "architectures": ["Phi3ForCausalLM"]}
        rows = [{"text": "First?", "label": "ANSWERABLE"},
                {"text": "Second?", "label": "UNANSWERABLE"}]
        answers = [("ANSWERABLE", [1, 100265], False), ("UNANSWERABLE", [2, 100265], False)]
        probes = [evaluator.shared.PROMPT.format(question=question) for question, _ in evaluator.SMOKE_PROBES]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = evaluator.output_path(root, "umwp", "run_01")
            with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
                    patch.object(sys, "argv", ["evaluator", "--dataset", "umwp", "--run-id", "run_01"]), \
                    patch.object(evaluator, "ROOT", root), \
                    patch.object(evaluator, "MODEL_PATH", root / "models/phi-4"), \
                    patch.object(evaluator, "load_config", return_value=config), \
                    patch.object(evaluator, "load_generator") as load, \
                    patch.object(evaluator, "transformers_distribution_url", return_value=None), \
                    patch.object(evaluator.shared, "load_dataset", return_value=(rows, "dataset hash")), \
                    patch.object(evaluator.shared, "model_fingerprint", return_value="weights hash"), \
                    patch.object(evaluator.signal, "signal"), patch("builtins.print"):
                failed = load.return_value = Mock(side_effect=[answers[0], ("UNANSWERABLE", [2] * 16, True)])
                self.assertEqual(evaluator.main(), 1)
                self.assertEqual([call.args[0] for call in failed.call_args_list], probes)
                self.assertFalse(path.exists())

                interrupted = load.return_value = Mock(side_effect=[*answers, answers[0], RuntimeError("interrupted")])
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    evaluator.main()
                self.assertEqual([call.args[0] for call in interrupted.call_args_list[:2]], probes)
                self.assertTrue((path / "checkpoint.sqlite3").is_file())

                resumed = load.return_value = Mock(side_effect=[*answers, answers[1]])
                self.assertEqual(evaluator.main(), 0)
                self.assertEqual([call.args[0] for call in resumed.call_args_list],
                                 [*probes, evaluator.shared.PROMPT.format(question="Second?")])
                self.assertEqual(load.call_count, 3)
            self.assertTrue((path / "predictions.csv").is_file())
            self.assertTrue((path / "confusion_matrix.csv").is_file())
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["run"]["model_id"], "microsoft/phi-4")
            self.assertEqual(metrics["run"]["max_new_tokens"], 16)

