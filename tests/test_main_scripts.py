import pytest
import subprocess
import sys
import glob
import os

# Append scripts paths
# sys.path.append(os.getcwd() + '/example_scripts/FLFMicroscope/')
# print(os.getcwd())

script_paths = ['example_scripts/FLFMicroscope/main*.py', 'example_scripts/WFMicroscope/*.py']

files_to_test = []
# Gather main files to test
for script in script_paths:
    files = glob.glob(script)
    for f in files:
        files_to_test.append(f.replace('/','.')[:-3])

# Create automathic tests, by importing them
@pytest.mark.parametrize("script_to_test", files_to_test)
def test_LF_scripts(script_to_test):
    i = __import__(script_to_test)




# def test_FLFM():
#     import example_scripts.FLFMicroscope.main_RichardsonLucy_example

# test_FLFM()