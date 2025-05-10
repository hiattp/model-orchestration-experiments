import datetime
import pathlib

DIVIDER = f"\n{'=' * 200}\n"


def make_experiment_run(experiment_name: str, save_results=False) -> tuple[str, str]:
    experiment_run_datetime = datetime.datetime.now()
    experiment_run = datetime.datetime.strftime(experiment_run_datetime, "%Y%m%d%H%M%S")
    destination = None

    print(
        f"{DIVIDER}Running {experiment_name} experiment, {experiment_run} run at {experiment_run_datetime}{DIVIDER}"
    )

    if save_results:
        destination = get_experiment_run_destination(experiment_name, experiment_run)
        print(
            f'Saving results of {experiment_name} experiment, {experiment_run} run at "{destination}"...'
        )
    else:
        print(
            f"No results will be saved for {experiment_name} experiment, {experiment_run}..."
        )

    return (experiment_run, destination)


def end_experiment_run(experiment_name: str, experiment_run: str):
    print(
        f"{DIVIDER}Ending {experiment_name} experiment, {experiment_run} run at {datetime.datetime.now()}{DIVIDER}"
    )
    # potentially other clean-up


def get_experiment_run_destination(experiment_name: str, experiment_run: str) -> str:
    destination = f"data/out/{experiment_name.lower()}/{str(experiment_run)}"
    pathlib.Path(destination).mkdir(parents=True, exist_ok=True)

    return destination


# Helper: check if two bounding boxes overlap (simple polygon intersection)
def bboxes_overlap(bbox1, bbox2):
    # bbox: [x1, y1, x2, y2, x3, y3, x4, y4] (rectangle corners)
    # We'll use min/max for a simple rectangle overlap
    x1s = bbox1[::2]
    y1s = bbox1[1::2]
    x2s = bbox2[::2]
    y2s = bbox2[1::2]
    min_x1, max_x1 = min(x1s), max(x1s)
    min_y1, max_y1 = min(y1s), max(y1s)
    min_x2, max_x2 = min(x2s), max(x2s)
    min_y2, max_y2 = min(y2s), max(y2s)
    return not (
        max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1
    )
