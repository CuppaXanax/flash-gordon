import importlib.util
import os
from pathlib import Path
import shutil
import struct
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("mtp_vocab_pack",
    Path(__file__).parents[1]/"tools"/"mtp_vocab_pack.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VocabPackTest(unittest.TestCase):
    def setUp(self):
        self.root=Path(".build")/f"mtp-vocab-pack-{os.getpid()}"
        self.root.mkdir()
        self.source=self.root/"source.q8"
        self.source.write_bytes(b"source matrix")
        self.output=self.root/"subset.fgdv"

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_header_sorted_ids_and_source_preserved(self):
        with patch.object(MODULE,"RAW_BYTES",self.source.stat().st_size):
            report=MODULE.pack(self.source,self.output,[19,2,100],"ab"*32)
        raw=self.output.read_bytes()
        self.assertEqual(raw[:8],b"FGDVOC1\0")
        self.assertEqual(struct.unpack("<I",raw[8:12]),(3,))
        self.assertEqual(raw[12:44],bytes.fromhex("ab"*32))
        self.assertEqual(struct.unpack("<3I",raw[44:56]),(2,19,100))
        self.assertEqual(raw[56:],self.source.read_bytes())
        self.assertEqual(report["rows"],3)
        with self.assertRaises(FileExistsError),patch.object(MODULE,"RAW_BYTES",13):
            MODULE.pack(self.source,self.output,[1],"ab"*32)

    def test_invalid_selection_parent_and_source_fail(self):
        for ids in ([],[1,1],[-1],[248320],[True],[1.2]):
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                MODULE.pack(self.source,self.output,ids,"ab"*32)
        for parent in ("zz","ab"):
            with self.subTest(parent=parent),self.assertRaises(ValueError):
                MODULE.pack(self.source,self.output,[1],parent)
        with self.assertRaises(ValueError):
            MODULE.pack(self.source,self.output,[1],"ab"*32)
        self.assertFalse(self.output.exists())
