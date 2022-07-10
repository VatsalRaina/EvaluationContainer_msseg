from pathlib import Path
import json


def load_predictions_json(fname: Path):

    cases = {}

    with open(fname, "r") as f:
        entries = json.load(f)

    if isinstance(entries, float):
        raise TypeError(f"entries of type float for file: {fname}")

    for e in entries:
        # Find case name through input file name
        inputs = e["inputs"]
        name = None
        for input in inputs:
            if input["interface"]["slug"] == "generic-medical-image":
                name = str(input["image"]["name"])
                break  # expecting only a single input
        if name is None:
            raise ValueError(f"No filename found for entry: {e}")

        entry = {"name": name}

        # Find output value for this case
        outputs = e["outputs"]

        for output in outputs:
            if output["interface"]["slug"] == "generic-medical-image":
                pk = output["image"]["pk"]
                if ".mha" not in pk:
                    pk += ".mha"
                cases[pk] = name

    return cases


def test():
    mapping_dict = load_predictions_json(Path("test/predictions.json"))
    return mapping_dict


if __name__ == "__main__":

    mapping_dict = test()
