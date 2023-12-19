import ray

# import mlflow
from datetime import datetime
from launcher import GAZLauncher
from gaz_singleplayer.config_msa import Config

CUDA_LAUNCH_BLOCKING = 1

if __name__ == "__main__":
    current_dateTime = datetime.now()
    # mlflow.set_tracking_uri()
    exp_name = f"msa-experiment-{str(current_dateTime)}"
    # experiment_id = mlflow.get_experiment_by_name(exp_name)
    # if experiment_id is None:
    #     experiment_id = mlflow.create_experiment(exp_name)
    # else:
    #     experiment_id = experiment_id.experiment_id
    # with mlflow.start_run(experiment_id=experiment_id, run_name="run_name") as run:
    #     run_id = run.info.run_id
    #     mlflow.set_experiment(experiment_name=exp_name)

    config = Config(
        save_model=True,
        exp_name=exp_name,
        run_id=0,
        training_games=30000,
        reward_method="SumOfPairs+10*TotalColumn",
        reward_values={"GG": 0, "GL": 0, "LL": 1, "LDL": 0},
    )

    gaz = GAZLauncher(config)
    gaz.setup_workers()

    if config.training_games > 0:
        print(f"Starting Training...")
        gaz.train()
    else:
        if not config.checkpoint_pth:
            print("WARNING: Testing mode, but no checkpoint to load was specified.")
        gaz.test()

    ray.shutdown()
    # mlflow.end_run()
