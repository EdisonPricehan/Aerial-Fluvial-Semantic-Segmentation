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


if __name__ == '__main__':
    api = wandb.Api()
    # specify the run to update
    run = api.run("edisonprincehan/baseline_test/3l40t08l")

    update_config(run, "trainset", "pretraining")
