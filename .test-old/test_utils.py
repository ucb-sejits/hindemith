import unittest
from hindemith.utils import unique_name, clamp

__author__ = 'leonardtruong'


class TestUtils(unittest.TestCase):
    def test_unique_name(self):
        self.assertNotEqual(unique_name(), unique_name())

    def test_clamp_max(self):
        self.assertEqual(clamp(20, 0, 15), 15)

    def test_clamp_min(self):
        self.assertEqual(clamp(0, 5, 15), 5)

    def test_clamp_noop(self):
        self.assertEqual(clamp(10, 5, 15), 10)
