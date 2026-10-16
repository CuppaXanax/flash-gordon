import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import struct
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("mtp_vocab_inspect",
    Path(__file__).parents[1]/"tools"/"mtp_vocab_inspect.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VocabInspectTest(unittest.TestCase):
    def setUp(self):
        self.root=Path(".build")/f"mtp-vocab-inspect-{os.getpid()}"
        self.root.mkdir()
        self.file=self.root/"head.q8"
        self.source=b"authentic raw source"
        self.sha=hashlib.sha256(self.source).hexdigest()
        self.parent="ab"*32

    def tearDown(self):
        shutil.rmtree(self.root)

    def run_inspect(self, data):
        self.file.write_bytes(data)
        with patch.object(MODULE,"RAW_BYTES",len(self.source)):
            return MODULE.inspect(self.file,self.sha,self.parent)

    def test_raw_and_container(self):
        self.assertEqual(self.run_inspect(self.source)["format"],"raw-q8")
        data=b"FGDVOC1\0"+struct.pack("<I",3)+bytes.fromhex(self.parent)+struct.pack("<3I",1,17,248319)+self.source
        report=self.run_inspect(data)
        self.assertEqual(report["rows"],3)
        self.assertEqual(report["sha256"],hashlib.sha256(data).hexdigest())
        self.assertTrue(report["full_source_authenticated"])

    def test_bad_source_parent_ids_size(self):
        prefix=b"FGDVOC1\0"+struct.pack("<I",3)+bytes.fromhex(self.parent)
        for data in (
            self.source[:-1]+b"x",
            prefix+struct.pack("<3I",1,17,17)+self.source,
            prefix+struct.pack("<3I",1,248320,248321)+self.source,
            prefix+struct.pack("<3I",17,1,50)+self.source,
            prefix+struct.pack("<3I",1,17,50)+self.source[:-1],
            prefix+struct.pack("<3I",1,17,50)+self.source+b"x",
            prefix+struct.pack("<3I",1,17,50)+self.source[:-1]+b"x",
            b"FGDVOC1\0"+struct.pack("<I",0)+bytes.fromhex(self.parent)+self.source,
            prefix[:12]+bytes.fromhex("cd"*32)+struct.pack("<3I",1,17,50)+self.source,
        ):
            with self.subTest(data=data),self.assertRaises(ValueError):
                self.run_inspect(data)
