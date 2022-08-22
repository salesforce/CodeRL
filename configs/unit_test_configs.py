#
# '''
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# '''#
import argparse

parser = argparse.ArgumentParser(description="Testing generated programs with unit testing")

parser.add_argument("-t","--test_path", default="data/APPS/test/", type=str, help="Path to test samples")
parser.add_argument("--output_path", type=str, help="Path to output test results")
parser.add_argument("--code_path", type=str, help='Path to generated programs') 

parser.add_argument("-i", "--index", default=-1, type=int, help='specific sample index to be tested against unit tests')
parser.add_argument("-d", "--debug", action="store_true", help='test in debugging mode with printout messages')
parser.add_argument('--max_tests', type=int, default=-1, help='Filter for test samples by maximum number of unit tests') 
parser.add_argument('--example_tests', type=int, default=0, help='0: run hidden unit tests; 1: run example unit tests')

args = parser.parse_args()

