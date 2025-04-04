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
