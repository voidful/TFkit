import os
import unittest

from transformers import AutoTokenizer

import tfkit


class TestPackage(unittest.TestCase):
    def testImport(self):
        path = os.path.dirname(tfkit.__file__)
        print(path)
        tfkit.task
        tfkit.utility
