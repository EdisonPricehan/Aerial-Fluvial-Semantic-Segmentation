#!E:\anaconda/python

import wandb


def update_config(run, key, new_conf):
    if key in run.config.keys():
        old_conf = run.config[key]
        run.config[key] = new_conf
        run.update()
        print(f"Updated {key} from {old_conf} to {new_conf}.")
    else:
        print(f"{key} is not in the config keys.")


def add_config(run, key, new_conf):
    if key in run.config.keys():
        print(f"{key} is already in the config keys.")
    else:
        run.config[key] = new_conf
        run.update()
        print(f"Added {key} with {new_conf}.")


if __name__ == '__main__':
    api = wandb.Api()
    # specify the run to update
    run = api.run("edisonprincehan/baseline_test/qtnzqus4")

    # update existing config
    # update_config(run, "trainset", "pretraining")

    # add new config
    # add_config(run, "Params Mb", 32.5)
    # add_config(run, "GFLOPs", 28.35)

