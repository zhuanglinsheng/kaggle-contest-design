import unittest

from pipeline.jk9_fit import plan


class JK9InputTests(unittest.TestCase):
    def test_all_main_frozen_inputs_are_self_contained_and_hash_matched(self):
        full = plan('main', 'full')
        early = plan('main', 'early')
        self.assertEqual(len(full), 73)
        self.assertEqual({r['contest_id'] for r in full}, {r['contest_id'] for r in early})

    def test_single_case_and_unknown_case(self):
        self.assertEqual([r['contest_id'] for r in plan('main', 'full', 2445)], [2445])
        with self.assertRaises(ValueError):
            plan('main', 'full', -1)
