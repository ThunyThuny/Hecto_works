"""V4 dry-run preserves the frozen A and never reaches model execution."""
from pathlib import Path
from unittest.mock import patch
import unittest
import test_b_experiment_script as legacy

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_eval_b_v4_gpu5.sh"


class V4ScriptTests(unittest.TestCase):
    def test_old_script_is_separate_and_new_candidate_stages_are_ordered(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("b-synthetic-v4-gpu5-", text)
        self.assertIn("--input-version target_focus_v2 --mask-class-weight 3.0", text)
        self.assertIn("--label-smoothing 0.02 --early-stopping-patience 1", text)
        self.assertIn("--max-added-leak-rate 0.0", text)
        self.assertIn("tokenizer_input_audit", text)
        gate = text.index('if [[ "$KPII_CHECK_ONLY" == true ]]')
        self.assertLess(gate, text.index("source scripts/activate_gpu5.sh"))
        self.assertLess(text.index("kpii predict-a"), text.index("kpii refine-a-candidates"))
        self.assertLess(text.index("kpii refine-a-candidates"), text.index("kpii score-b"))
        self.assertIn("kpii diagnose-b-classification", text)
        self.assertIn("a-raw-candidates.jsonl", text)


@unittest.skipUnless(legacy.os.name == 'posix' and legacy.shutil.which('bash'), 'POSIX bash subprocess fixture')
class V4ScriptExecutionTests(legacy.ExperimentScriptExecutionTests):
    def setUp(self):
        patcher = patch.object(legacy, "SCRIPT", SCRIPT)
        patcher.start()
        self.addCleanup(patcher.stop)
        super().setUp()
