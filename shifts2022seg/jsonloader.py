from pathlib import Path
import json


def load_predictions_json(fname: Path):

    cases = {}

    with open(fname, "r") as f:
        entries = json.load(f)

    if isinstance(entries, float):
        raise TypeError(f"entries of type float for file: {fname}")

    for e in entries:
        pk = e["pk"]
        # Find case name through input file name
        inputs = e["inputs"]
        name = None
        for input in inputs:
            if input["interface"]["slug"] == "brain-mri":
                name = str(input["image"]["name"])
                break  # expecting only a single input

        cases[pk] = name

    return cases


def test():
    mapping_dict = load_predictions_json(Path("test/predictions.json"))
    return mapping_dict


if __name__ == "__main__":

    mapping_dict = test()
