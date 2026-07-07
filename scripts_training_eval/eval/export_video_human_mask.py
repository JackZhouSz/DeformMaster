import argparse
import os
import glob
import json

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/data_process/data_config.csv")
parser.add_argument("--base-path", default="./data/different_types")
parser.add_argument("--output-path", default="./data/different_types_human_mask")
args = parser.parse_args()

base_path = args.base_path
output_path = args.output_path


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


import csv

existDir(output_path)

with open(args.config, newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        case_name = row[0]

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        print(f"Processing {case_name}!!!!!!!!!!!!!!!")
        existDir(f"{output_path}/{case_name}")

        # Detect number of cameras
        with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
            meta = json.load(f)
        camera_num = len(meta["intrinsics"])

        TEXT_PROMPT = "human"

        for camera_idx in range(camera_num):
            print(f"Processing {case_name} camera {camera_idx}")
            os.system(
                f"python ./data_process/segment_util_video.py --output_path {output_path}/{case_name} --base_path {base_path} --case_name {case_name} --TEXT_PROMPT '{TEXT_PROMPT}' --camera_idx {camera_idx}"
            )
            os.system(f"rm -rf {base_path}/{case_name}/tmp_data")
