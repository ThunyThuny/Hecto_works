import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SingleGpuStaticTests(unittest.TestCase):
    def test_main_is_sequential(self):
        source = (ROOT / "main_gemma4_v2.py").read_text(encoding="utf-8")
        self.assertNotIn("ThreadPoolExecutor", source)
        self.assertLess(source.index("load_whisper_model()"), source.index("load_diarization_model()"))
        self.assertLess(source.index("load_diarization_model()"), source.index("load_llm_model()"))

    def test_pipeline_targets_only_logical_gpu_zero(self):
        source = (ROOT / "pipeline_gemma4_v2.py").read_text(encoding="utf-8")
        self.assertNotIn("cuda:1", source)
        self.assertNotIn("cuda:2", source)
        self.assertNotIn("tensor_split", source)
        self.assertIn('torch.device("cuda:0")', source)

    def test_no_embedded_huggingface_token(self):
        for path in (ROOT / "pipeline_gemma4_v2.py", ROOT / "main_gemma4_v2.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
            self.assertFalse(any(value.startswith("hf_") for value in strings), path.name)


if __name__ == "__main__":
    unittest.main()
