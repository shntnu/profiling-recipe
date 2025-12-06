"""
Perform the profiling pipeline (defined in profile.py).
"""

# Modified from
# https://github.com/broadinstitute/profiling-resistance-mechanisms/blob/master/0.generate-profiles/generate-profiles.py

from multiprocessing import Pool, cpu_count
from utils import load_pipeline, create_directories
from profile import RunPipeline
import argparse

parser = argparse.ArgumentParser(description="Run the profiling pipeline")
parser.add_argument("--config", help="Config file")
parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")

args = parser.parse_args()

pipeline, profile_config = load_pipeline(config_file=args.config)


def process_plate(plate_args):
    """Process a single plate (aggregate, annotate, normalize steps)."""
    batch, plate, pipeline, profile_config = plate_args

    # Each process needs its own RunPipeline instance
    from utils import create_directories
    from profile import RunPipeline
    run_pipeline = RunPipeline(pipeline=pipeline, profile_config=profile_config)

    create_directories(batch=batch, plate=plate, pipeline=pipeline)

    if "aggregate" in pipeline:
        if pipeline["aggregate"]["perform"]:
            print(f"Now aggregating... plate: {plate}", flush=True)
            run_pipeline.pipeline_aggregate(batch=batch, plate=plate)

    if "annotate" in pipeline:
        if pipeline["annotate"]["perform"]:
            print(f"Now annotating... plate: {plate}", flush=True)
            run_pipeline.pipeline_annotate(batch=batch, plate=plate)

    if "normalize" in pipeline:
        if pipeline["normalize"]["perform"]:
            print(f"Now normalizing... plate: {plate}", flush=True)
            if pipeline["normalize"]["min_cells"] == 1:
                norm_samples = "all"
            else:
                norm_samples = f'Metadata_Object_Count >= {pipeline["normalize"]["min_cells"]}'
            run_pipeline.pipeline_normalize(
                batch=batch, plate=plate, steps=pipeline["normalize"], samples=norm_samples
            )

    if "normalize_negcon" in pipeline:
        if pipeline["normalize_negcon"]["perform"]:
            print(f"Now normalizing to negcon... plate: {plate}", flush=True)
            if pipeline["normalize_negcon"]["min_cells"] == 1:
                norm_negcon_samples = "Metadata_control_type == 'negcon'"
            else:
                norm_negcon_samples = f"Metadata_control_type == 'negcon' & Metadata_Object_Count >= {pipeline['normalize_negcon']['min_cells']}"
            run_pipeline.pipeline_normalize(
                batch=batch,
                plate=plate,
                steps=pipeline["normalize_negcon"],
                samples=norm_negcon_samples,
                suffix="negcon",
            )

    print(f"Completed plate: {plate}", flush=True)
    return plate


n_workers = args.workers or min(cpu_count(), 64)  # Cap at 64 to avoid overwhelming

for batch in profile_config:
    plates = list(profile_config[batch])
    print(f"Now processing... batch: {batch} ({len(plates)} plates with {n_workers} workers)")

    plate_args = [(batch, plate, pipeline, profile_config) for plate in plates]

    with Pool(processes=n_workers) as pool:
        pool.map(process_plate, plate_args)

# Batch-level operations after all plates are done
run_pipeline = RunPipeline(pipeline=pipeline, profile_config=profile_config)

if "feature_select" in pipeline:
    if pipeline["feature_select"]["perform"]:
        print(f"Now feature selecting... level: {pipeline['feature_select']['level']}")
        run_pipeline.pipeline_feature_select(steps=pipeline["feature_select"], min_cells = pipeline["feature_select"]["min_cells"])

if "feature_select_negcon" in pipeline:
    if pipeline["feature_select_negcon"]["perform"]:
        print(
            f"Now feature selecting negcon profiles... level: {pipeline['feature_select_negcon']['level']}"
        )
        run_pipeline.pipeline_feature_select(
            steps=pipeline["feature_select_negcon"], suffix="negcon", min_cells = pipeline["feature_select_negcon"]["min_cells"]
        )

if "quality_control" in pipeline:
    if pipeline["quality_control"]["perform"]:
        run_pipeline.pipeline_quality_control(operations=pipeline["quality_control"])
