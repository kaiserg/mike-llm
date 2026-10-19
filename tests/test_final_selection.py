import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('finalize_stage',Path(__file__).resolve().parents[1]/'scripts/finalize_stage.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class FinalSelectionTests(unittest.TestCase):
    def test_validation_loss_and_stable_ties(self):
        candidates=[{'id':'z','validation':{'mean_nll':2.}},
                    {'id':'b','validation':{'mean_nll':1.}},
                    {'id':'a','validation':{'mean_nll':1.}}]
        self.assertEqual(module.choose(candidates)['id'],'a')
        self.assertEqual(module.choose(list(reversed(candidates)))['id'],'a')
        with self.assertRaises(ValueError):module.choose([])

    def test_frozen_files_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'selection.json'
            module.write_new(path,{'frozen':True})
            with self.assertRaises(FileExistsError):module.write_new(path,{'frozen':False})
            self.assertEqual(json.loads(path.read_text()),{'frozen':True})

    def test_inconsistent_selection_rejected_before_test_or_model_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);selection=root/'selection.json'
            selection.write_text(json.dumps(dict(stage='validation-only frozen selection',test_content_read=False,
                candidates=[dict(id='a',validation=dict(mean_nll=1.))],selected=dict(id='b'))))
            args=SimpleNamespace(selection=selection,output=root/'test.json')
            with patch.object(module,'load_checkpoint',side_effect=AssertionError('model must not load')):
                with self.assertRaisesRegex(ValueError,'inconsistent'):module.evaluate(args)
            self.assertFalse(args.output.exists())
