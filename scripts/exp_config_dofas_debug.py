import subprocess

# Define all the experiments
########################-DOFA-########################
experiments = [

    # {
    #     "model": "croma_cls_s1",
    #     "dataset": "senbench_benv2s1_croma",
    #     "task": "classification",
    #     "batch_size": 64,
    #     "lr": 1,
    #     "epochs": 10,
    # },
    # {
    #     "model": "croma_cls_s1",
    #     "dataset": "senbench_eurosats1",
    #     "task": "classification",
    #     "batch_size": 64,
    #     "lr": 1,
    #     "epochs": 10,
    # },
    # {
    #     "model": "croma_cls",
    #     "dataset": "senbench_eurosats2_croma",
    #     "task": "classification",
    #     "batch_size": 64,
    #     "lr": 1,
    #     "epochs": 10,
    # },
    # {
    #     "model": "croma_seg_s1",
    #     "dataset": "senbench_dfc2020s1",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "lr": 0.001,
    #     "epochs": 10,
    # },
    # {
    #     "model": "croma_seg",
    #     "dataset": "senbench_dfc2020s2_croma",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "lr": 0.001,
    #     "epochs": 10,
    # },
    # {
    #     "model": "croma_seg",
    #     "dataset": "senbench_clouds2_croma",
    #     "task": "segmentation",
    #     "batch_size": 4,
    #     "lr": 0.001,
    #     "epochs": 10,
    # },
    # {
    #     "model": "croma_cls",
    #     "dataset": "senbench_so2sats2_croma",
    #     "task": "classification",
    #     "batch_size": 64,
    #     "lr": 1,
    #     "epochs": 10,
    # },
    {
        "model": "dofas_cls",
        "dataset": "senbench_benv2s1",
        "task": "classification",
        "batch_size": 64,
        "lr": 1,
        "epochs": 50,
    },


]

# Run each experiment
for exp in experiments:
    print(f"Running experiment: {exp['model']} on {exp['dataset']}")
    # exp["epochs"] = 1  # This is for debug
    if "warmup_epochs" not in exp.keys():
        exp["warmup_epochs"] = 0
    subprocess.run(
        [
            "bash",
            "scripts/run_dofas.sh",  # Path to the template script
            exp["model"],
            exp["dataset"],
            exp["task"],
            str(exp["batch_size"]),
            str(exp["lr"]),
            str(exp["epochs"]),
            str(exp["warmup_epochs"]),
        ],
        check=True,
    )
    print(f"Completed: {exp['model']} on {exp['dataset']}")
