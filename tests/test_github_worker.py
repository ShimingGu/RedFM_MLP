"""Defaults must target Cross-Matching without silently switching existing work."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import github_worker


class GitHubWorkerBranchTests(unittest.TestCase):
    def test_empty_push_choice_targets_cross_matching(self):
        with mock.patch('builtins.input', return_value=''):
            self.assertEqual(github_worker.choose_push_branch('main'), 'Cross-Matching')

    def test_explicit_push_branch_is_respected(self):
        with mock.patch('builtins.input', return_value='  experimental  '):
            self.assertEqual(github_worker.choose_push_branch('main'), 'experimental')

    def test_new_repository_initializes_on_cross_matching(self):
        with tempfile.TemporaryDirectory() as folder:
            root = github_worker.ensure_git_repo(Path(folder))
            self.assertEqual(github_worker.current_branch(root), 'Cross-Matching')

    def test_existing_repository_branch_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            github_worker.git(['init', '--initial-branch', 'main'], cwd=root)
            self.assertEqual(github_worker.ensure_git_repo(root), root.resolve())
            self.assertEqual(github_worker.current_branch(root), 'main')


if __name__ == '__main__':
    unittest.main()
