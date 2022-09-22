import os

def load_benchmark(filepath="data/nguyen.csv"):

    if filepath is None:
        filepath="data/nguyen_test.csv"

    with open(filepath, "r") as f:
        lines = f.readlines()


    return lines

