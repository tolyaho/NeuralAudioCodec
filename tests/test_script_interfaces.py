import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def run_help(script: str) -> str:
    proc = subprocess.run(
        [str(PYTHON), script, "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return proc.stdout


class ScriptInterfaceTest(unittest.TestCase):
    def test_training_scripts_use_hydra_help(self):
        for script in ("scripts/train.py", "scripts/train_gan.py"):
            out = run_help(script)
            self.assertIn("powered by Hydra", out)
            self.assertIn("Override anything in the config", out)

    def test_inference_script_uses_hydra_help(self):
        out = run_help("scripts/inference.py")
        self.assertIn("powered by Hydra", out)
        self.assertIn("checkpoint:", out)

    def test_root_entrypoints_are_not_public_interface(self):
        self.assertFalse((ROOT / "train.py").exists())
        self.assertFalse((ROOT / "inference.py").exists())

    def test_training_scripts_use_template_comet_writer(self):
        for script in ("scripts/train.py", "scripts/train_gan.py"):
            text = (ROOT / script).read_text(encoding="utf-8")
            self.assertIn("from src.logger import CometMLWriter", text)
            self.assertNotIn("from comet_ml import Experiment", text)


if __name__ == "__main__":
    unittest.main()
