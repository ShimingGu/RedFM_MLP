"""Storage restoration must never replace user data or partially link bad archives."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/link_physics_storage.py'


class PhysicsStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / 'checkout'
        self.storage = Path(self.temp.name) / 'archive'
        (self.workspace / 'configs').mkdir(parents=True)
        self.records = []
        for name, content in [('first.npz', b'first array'), ('second.pt', b'second checkpoint')]:
            relative = Path('reports/stage4/test') / name
            path = self.storage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            self.records.append({'path': str(relative), 'bytes': len(content),
                                 'sha256': hashlib.sha256(content).hexdigest()})
        self.manifest = self.workspace / 'configs/physics_artifact_storage.json'
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text(json.dumps({'schema_version': 1,
            'storage_root': str(self.storage), 'file_count': len(self.records),
            'files': self.records}))

    def run_helper(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), '--workspace', str(self.workspace),
                               *args], capture_output=True, text=True)

    def test_restore_is_idempotent_and_read_only_check_passes(self):
        self.assertEqual(self.run_helper('--verify').returncode, 0)
        for record in self.records:
            link = self.workspace / record['path']
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), self.storage / record['path'])
        self.assertEqual(self.run_helper('--verify').returncode, 0)
        self.assertEqual(self.run_helper('--check-only', '--verify').returncode, 0)

    def test_checksum_failure_does_not_partially_create_links(self):
        self.records[-1]['sha256'] = '0' * 64
        self.save_manifest()
        self.assertNotEqual(self.run_helper('--verify').returncode, 0)
        self.assertFalse((self.workspace / self.records[0]['path']).exists())

    def test_existing_file_is_never_overwritten(self):
        existing = self.workspace / self.records[-1]['path']
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b'local work')
        self.assertNotEqual(self.run_helper('--verify').returncode, 0)
        self.assertEqual(existing.read_bytes(), b'local work')
        self.assertFalse((self.workspace / self.records[0]['path']).exists())

    def test_missing_link_check_does_not_create_anything(self):
        self.assertNotEqual(self.run_helper('--check-only').returncode, 0)
        self.assertFalse((self.workspace / 'reports').exists())

    def test_parent_traversal_is_rejected(self):
        self.records[0]['path'] = '../outside.npz'
        self.save_manifest()
        self.assertNotEqual(self.run_helper().returncode, 0)
        self.assertFalse((self.workspace.parent / 'outside.npz').exists())


if __name__ == '__main__':
    unittest.main()
