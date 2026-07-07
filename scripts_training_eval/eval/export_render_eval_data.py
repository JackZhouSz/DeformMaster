import argparse
import os
import csv
import json

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/data_process/data_config.csv")
parser.add_argument("--base-path", default="./data/different_types")
parser.add_argument("--output-path", default="./data/render_eval_data")
args = parser.parse_args()

base_path = args.base_path
output_path = args.output_path
CONTROLLER_NAME = "hand"


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


existDir(output_path)

with open(args.config, newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        # Detect number of cameras
        with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
            meta = json.load(f)
        num_cam = len(meta["intrinsics"])

        # Create the directory for the case
        existDir(f"{output_path}/{case_name}")
        existDir(f"{output_path}/{case_name}/mask")

        # Copy color directory
        os.system(
            f"cp -r {base_path}/{case_name}/color {output_path}/{case_name}/"
        )

        for i in range(num_cam):
            # Copy only the object mask image
            with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
                data = json.load(f)
            obj_ids = [int(key) for key, value in data.items()
                       if value != CONTROLLER_NAME]

            existDir(f"{output_path}/{case_name}/mask/{i}")
            for oid in obj_ids:
                os.system(f"cp -r {base_path}/{case_name}/mask/{i}/{oid}/* {output_path}/{case_name}/mask/{i}/")

        # Copy the split.json
        os.system(f"cp {base_path}/{case_name}/split.json {output_path}/{case_name}/")
