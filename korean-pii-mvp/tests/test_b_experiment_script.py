"""CPU-only wrapper boundary checks; fake commands never initialize a model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'train_eval_b_synthetic_experiment_gpu5.sh'


class ExperimentScriptStructureTests(unittest.TestCase):
    def test_experiment_mode_reaches_all_review_gates(self):
        source = SCRIPT.read_text(encoding='utf-8')
        for command in ('preflight-b', 'train-b', 'calibrate-b', 'evaluate-ab'):
            command_body = source.split('kpii ' + command + ' \\\n', 1)[1].split('\n\n', 1)[0]
            self.assertIn('--synthetic-experiment-dir "$KPII_PACKET_DIR"', command_body)
        self.assertNotIn('"review_status"] = "APPROVED"', source)
        self.assertIn("'review_status': 'PENDING'", source)

    def test_check_only_precedes_gpu_selection_and_run_creation(self):
        source = SCRIPT.read_text(encoding='utf-8')
        gate = source.index('if [[ "$KPII_CHECK_ONLY" == true ]]')
        self.assertLess(gate, source.index('source scripts/activate_gpu5.sh'))
        self.assertLess(gate, source.index('mkdir -- "$KPII_RUN_DIR"'))
        self.assertIn('trap kpii_finish EXIT', source)
        self.assertIn('KPII_A_LOCK_SHA256', source)
        self.assertIn("KPII_A_MANIFEST_SHA256='87fd30cf", source)


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'POSIX bash subprocess fixture')
class ExperimentScriptExecutionTests(unittest.TestCase):
    """Execute only --check-only with all Python validations mocked at process edge."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.tmp.name).resolve()
        self.project = self.root / 'project'
        scripts = self.project / 'scripts'
        scripts.mkdir(parents=True)
        self.packet = self.root / 'packet'
        self.packet.mkdir()
        for name in ('proposals/train.jsonl', 'proposals/valid.jsonl', 'proposals/test.jsonl',
                     'review/reviews.jsonl', 'generation_manifest.json', 'automated_audit.json',
                     'a_checkpoint.lock.json'):
            path = self.packet / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}\n', encoding='utf-8')
        self.model = self.root / 'fixed-a'
        self.model.mkdir()
        manifest = self.model / 'training_manifest.json'
        manifest.write_text('{}\n', encoding='utf-8')
        self.calls = self.root / 'calls.jsonl'
        self.activation = self.root / 'gpu-activation'
        (scripts / 'activate_gpu5.sh').write_text('printf selected > "' + str(self.activation) + '"\nreturn 93\n')
        self.python = self.root / 'fake-python'
        self.python.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
with open(os.environ['KPII_TEST_CALLS'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == '-':
    sys.stdin.read()
    if os.environ.get('KPII_TEST_PREFLIGHT_FAIL') == '1':
        raise SystemExit(72)
print(json.dumps({'status': 'validated_mock_process_edge', 'training_performed': False}))
''', encoding='utf-8')
        self.python.chmod(0o700)
        source = SCRIPT.read_text(encoding='utf-8')
        source = source.replace("KPII_A_MODEL_DIR='/workspace/korean-pii-mvp/runs/a-native-gpu5-20261019_020722/best'",
                                "KPII_A_MODEL_DIR='" + str(self.model) + "'")
        source = source.replace("KPII_PYTHON='/workspace/kpii-env/bin/python'", "KPII_PYTHON='" + str(self.python) + "'")
        source = source.replace("KPII_A_MANIFEST_SHA256='87fd30cf17ee1afe5ea1d995c9f511bc28d11e5935efdc80750c0a30dd56d591'",
                                "KPII_A_MANIFEST_SHA256='" + hashlib.sha256(manifest.read_bytes()).hexdigest() + "'")
        self.script = scripts / SCRIPT.name
        self.script.write_text(source, encoding='utf-8')
        self.run = self.root / 'new-run'

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self, *, fail_preflight=False, run=None):
        env = dict(os.environ, KPII_TEST_CALLS=str(self.calls))
        if fail_preflight:
            env['KPII_TEST_PREFLIGHT_FAIL'] = '1'
        return subprocess.run(['bash', str(self.script), '--check-only', str(self.packet), str(run or self.run)],
                              text=True, capture_output=True, env=env, timeout=20)

    def test_check_only_validates_without_gpu_output_or_source_edits(self):
        before = {str(p.relative_to(self.packet)): p.read_bytes() for p in self.packet.rglob('*') if p.is_file()}
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('Check-only completed', completed.stdout)
        self.assertFalse(self.activation.exists())
        self.assertFalse(self.run.exists())
        self.assertEqual(before, {str(p.relative_to(self.packet)): p.read_bytes() for p in self.packet.rglob('*') if p.is_file()})
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(sum('verify-a-lock' in call for call in calls), 2)
        self.assertEqual(sum(call[0] == '-' for call in calls), 1)

    def test_validation_failure_stops_without_gpu_or_run_and_still_verifies_a(self):
        completed = self.invoke(fail_preflight=True)
        self.assertEqual(completed.returncode, 72, completed.stderr)
        self.assertFalse(self.activation.exists())
        self.assertFalse(self.run.exists())
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(sum('verify-a-lock' in call for call in calls), 2)

    def test_existing_or_nested_output_rejected_before_python_or_gpu(self):
        self.run.mkdir()
        for target in (self.run, self.packet / 'new-child', self.model / 'new-child'):
            with self.subTest(target=target):
                completed = self.invoke(run=target)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertFalse(self.activation.exists())
                self.assertFalse(self.calls.exists())


if __name__ == '__main__':
    unittest.main()
