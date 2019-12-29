import unittest

from transformers import AutoTokenizer

import tfkit
import os

class TestPackage(unittest.TestCase):

    def testImport(self):
        path = os.path.dirname(tfkit.__file__)
        print(path)
        tfkit.tag
        tfkit.utility
        tfkit.gen_onebyone
        tfkit.gen_once
        tfkit.classifier
        tfkit.tag
