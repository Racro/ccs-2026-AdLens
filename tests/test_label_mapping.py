"""Regression test: deceptive-claim votes are counted as positive.

The CLI violation name is `deceptive_claim`, but the classifier prompt answers
MISLEADING. The ranked/standalone path used to derive the positive label with
violation.upper() ("DECEPTIVE_CLAIM"), so two MISLEADING votes became SAFE.

No models are called: classify_top_n is replaced by canned votes, and torch /
sentence_transformers are stubbed when they are not installed.

    python -m unittest discover tests
"""
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

DETECTION = Path(__file__).resolve().parent.parent / "detection"
sys.path.insert(0, str(DETECTION))

for name in ("torch", "sentence_transformers", "tqdm"):
    try:
        importlib.import_module(name)
    except ImportError:
        stub = types.ModuleType(name)
        stub.SentenceTransformer = object
        stub.util = types.SimpleNamespace()
        stub.tqdm = lambda x=None, **k: x
        sys.modules[name] = stub

import ensemble  # noqa: E402
import llm_judge  # noqa: E402


def _ranked(n=2):
    return [{"id": f"ad{i}", "score": 0.9} for i in range(n)]


class EnsembleMajorityVote(unittest.TestCase):
    def _run(self, violation, label):
        def fake_classify(candidates, *a, **k):
            return [{"id": c["id"], "label": label, "reason": ""} for c in candidates]
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(ensemble, "classify_top_n", fake_classify), \
             mock.patch.object(ensemble, "_unload_llm", lambda *a, **k: None):
            return ensemble.classify_top_n_ensemble(
                _ranked(), violation, ["m1", "m2"], "cpu", Path(d), n=2)

    def test_deceptive_claim_unanimous_misleading_is_positive(self):
        out = self._run("deceptive_claim", "MISLEADING")
        self.assertEqual({r["ensemble_label"] for r in out}, {"MISLEADING"})

    def test_scareware_unanimous_is_positive(self):
        out = self._run("scareware", "SCAREWARE")
        self.assertEqual({r["ensemble_label"] for r in out}, {"SCAREWARE"})

    def test_safe_stays_safe(self):
        out = self._run("deceptive_claim", "SAFE")
        self.assertEqual({r["ensemble_label"] for r in out}, {"SAFE"})


class JudgeCollation(unittest.TestCase):
    def _collate(self, violation, qlabel, glabel):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for key, lbl in (("qwen35_9b", qlabel), ("gemma_3_12b_it", glabel)):
                rec = [{"id": "ad0", "sim_rank": 1, "label": lbl, "reason": ""}]
                (d / f"{violation}_x_{key}_ranked_top1-1_classified.json").write_text(json.dumps(rec))
            out = llm_judge.build_ensemble_collated(d, violation)
            return json.loads(out.read_text())[0]["ensemble_label"]

    def test_deceptive_claim_agreement_is_positive(self):
        self.assertEqual(self._collate("deceptive_claim", "MISLEADING", "MISLEADING"), "MISLEADING")

    def test_disagreement_is_split(self):
        self.assertEqual(self._collate("deceptive_claim", "MISLEADING", "SAFE"), "SPLIT")

    def test_judge_prompt_names_classifier_label(self):
        rec = {"per_model": {}}
        self.assertIn("One model says MISLEADING", llm_judge.build_user_prompt(rec, "deceptive_claim"))


if __name__ == "__main__":
    unittest.main()
