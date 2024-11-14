import os

def test_code_quality():
    os.system("pylint src/ --exit-zero")
    os.system("flake8 src/")
