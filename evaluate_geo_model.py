"""
Steps 7 and 8 of computervision doc/train_geo_model_guide.md for a grid training
run: thresholds, then the taxon range evaluation, both from inatVisionAPI.

Paths come from the run's config: dataset_dir, experiment_dir and the eval
section. The model is the newest saved_model.h5 under experiment_dir, and
results go to an eval/ folder next to it.

Run on calculon as inaturalist, with geomodel-env activated:
    python3 evaluate_geo_model.py --config_file config_X_YY_grid.yml
"""

import glob
import os
import subprocess
import sys

import click
import yaml


def fail(message):
    sys.exit(f"ERROR: {message}")


def run(script, vision_api_dir, args):
    print(f"\n== {script}", flush=True)
    result = subprocess.run([sys.executable, script, *args], cwd=vision_api_dir)
    if result.returncode != 0:
        fail(f"{script} failed (exit {result.returncode})")


@click.command()
@click.option("--config_file", required=True)
@click.option("--vision_api_dir", default="/disk/mnt/data/repos/inatVisionAPI")
@click.option("--stop_after", type=int, help="only process the first N taxa")
def main(config_file, vision_api_dir, stop_after):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    eval_config = config.get("eval") or {}

    dataset_dir = config.get("dataset_dir")
    experiment_dir = config.get("experiment_dir")
    tre_dir = eval_config.get("taxon_range_eval_dir")
    elevation_tif = eval_config.get("elevation_tif")
    # not the training h3_resolution
    h3_resolution = eval_config.get("h3_resolution")

    for key, value in [("dataset_dir", dataset_dir), ("experiment_dir", experiment_dir),
                       ("eval.elevation_tif", elevation_tif),
                       ("eval.h3_resolution", h3_resolution)]:
        if value in (None, ""):
            fail(f"{key} is not set in {config_file}")

    models = glob.glob(os.path.join(experiment_dir, "*", "saved_model.h5"))
    if not models:
        fail(f"no saved_model.h5 under {experiment_dir} -- has the training finished?")
    model = max(models, key=os.path.getmtime)
    out_dir = os.path.join(os.path.dirname(model), "eval")

    taxonomy = os.path.join(dataset_dir, "taxonomy.csv")
    spatial_data = os.path.join(dataset_dir, "spatial_data.parquet")
    needed = [elevation_tif, taxonomy, spatial_data]
    if tre_dir:
        recalls = os.path.join(tre_dir, "taxon_range_recalls.csv")
        indices = os.path.join(tre_dir, "taxon_range_csvs")
        needed += [recalls, indices]
    for path in needed:
        if not os.path.exists(path):
            fail(f"missing: {path}")

    print(f"  model          {model}")
    print(f"  export         {dataset_dir}")
    print(f"  range eval     {tre_dir or 'none -- the range evaluation will be skipped'}")
    print(f"  elevation      {elevation_tif}")
    print(f"  h3 resolution  {h3_resolution}")
    print(f"  output         {out_dir}")

    os.makedirs(out_dir, exist_ok=True)
    stop = ["--stop_after", str(stop_after)] if stop_after else []

    run("generate_thresholds.py", vision_api_dir, [
        "--elevation", elevation_tif,
        "--model", model,
        "--taxonomy", taxonomy,
        "--train_spatial_data", spatial_data,
        "--output_dir", out_dir,
        "--h3_resolution", str(h3_resolution),
        *stop,
    ])

    if not tre_dir:
        print(f"\nno eval.taxon_range_eval_dir in {config_file} -- range evaluation skipped")
        return

    run("taxon_range_evaluation.py", vision_api_dir, [
        "--elevation", elevation_tif,
        "--model", model,
        "--taxonomy", taxonomy,
        "--thresholds", os.path.join(out_dir, "thresholds.csv"),
        "--taxon_range_recalls", recalls,
        "--taxon_range_indicies", indices + "/",
        "--output_path", os.path.join(out_dir, "taxon_range_evaluation.csv"),
        *stop,
    ])

    print(f"\ndone: {out_dir}")


if __name__ == "__main__":
    main()
