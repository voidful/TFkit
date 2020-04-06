import unittest

import os


class TestEval(unittest.TestCase):

    def testHelp(self):
        result = os.system('tfkit-eval -h')
        assert (result == 0)

    def testEvalOnGen(self):
        result = os.system(
            'tfkit-eval --model ./cache/10.pt --valid ../demo_data/generate.csv --metric emf1 --print')
        print(result)
        result = os.system(
            'tfkit-eval --model ./cache/10.pt --valid ../demo_data/generate.csv --metric emf1 --print --beamsearch --outfile')
        print(result)
