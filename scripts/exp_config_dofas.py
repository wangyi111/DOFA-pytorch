import subprocess

# Define all the experiments
########################-DOFA-########################
experiments = [
    # {
    #     "model": "dofas_cls",
    #     "dataset": "geobench_eurosat",
    #     "task": "classification",
    #     "batch_size": 64,
    #     "lr": 0.05,
    #     "epochs": 50,
    # },
    # {
    #     "model": "dofas_seg",
    #     "dataset": "senbench_clouds2",
    #     "task": "segmentation",
    #     "batch_size": 8,
    #     "lr": 0.001,
    #     "epochs": 50,
    # },    
    # {
    #     "model": "dofas_cls",
    #     "dataset": "senbench_eurosats2",
    #     "task": "classification",
    #     "batch_size": 64,
    #     "lr": 0.05,
    #     "epochs": 50,
    # },   
    # {
    #     "model": "dofas_cls",
    #     "dataset": "senbench_eurosats1",
    #     "task": "classification",
    #     "batch_size": 64,
    #     "lr": 0.01,
    #     "epochs": 50,
    # },      
    # {
    #     "model": "dofas_cls",
    #     "dataset": "senbench_benv2s1",
    #     "task": "classification",
    #     "batch_size": 64,
    #     "lr": 0.01,
    #     "epochs": 50,
    # },
    # {
    #     "model": "dofas_cls",
    #     "dataset": "senbench_benv2s2",
    #     "task": "classification",
    #     "batch_size": 64,
    #     "lr": 0.01,
    #     "epochs": 50,
    # },
    # {
    #     "model": "dofas_cls",
    #     "dataset": "senbench_lc100s3cls",
    #     "task": "classification",
    #     "batch_size": 64,
    #     "lr": 0.01,
    #     "epochs": 50,
    # },
    # {
    #     "model": "dofas_seg",
    #     "dataset": "senbench_lc100s3seg",
    #     "task": "segmentation",
    #     "batch_size": 8,
    #     "lr": 0.001,
    #     "epochs": 10,
    # },
    # {
    #     "model": "dofas_reg",
    #     "dataset": "senbench_aqs5pno2",
    #     "task": "regression",
    #     "batch_size": 8,
    #     "lr": 0.001,
    #     "epochs": 10,
    # },
    {
        "model": "dofas_reg",
        "dataset": "senbench_aqs5po3",
        "task": "regression",
        "batch_size": 8,
        "lr": 0.001,
        "epochs": 10,
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
