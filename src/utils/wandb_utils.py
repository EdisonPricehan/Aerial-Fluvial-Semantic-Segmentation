#!E:\anaconda/python

import wandb


def update_config(run, key, old_conf, new_conf):
    if key in run.config.keys():
        if old_conf != '' and old_conf == run.config[key] or old_conf == '':
            run.config[key] = new_conf
            run.update()
            print(f"Updated {key} from {old_conf} to {new_conf}.")
            return True
        else:
            print(f"{old_conf} does not match the existing value {run.config[key]} of {key}.")
            return False
    else:
        print(f"{key} is not in the config keys.")
        return False


def add_config(run, key, new_conf):
    if key in run.config.keys():
        print(f"{key} is already in the config keys.")
        return False
    else:
        run.config[key] = new_conf
        run.update()
        print(f"Added {key} with {new_conf}.")
        return True


def update_project_config(runs, key, old_config, new_conf):
    for run in runs:
        if not update_config(run, key, old_config, new_conf):
            print(f"Failed to update {key} from {old_config} to {new_conf}.")
            continue
    print(f"Updated {key} in all runs from {old_config} to {new_conf}.")
    return True


if __name__ == '__main__':
    api = wandb.Api()
    # specify the run to update
    run = api.run("edisonprincehan/baseline_test/qtnzqus4")

    # update existing config
    # update_config(run, "trainset", "pretraining")

    # add new config
    # add_config(run, "Params Mb", 32.5)
    # add_config(run, "GFLOPs", 28.35)

    # update config of all runs in the project
    runs = api.runs("edisonprincehan/baseline_test")
    # update_project_config(runs, "arch", "DeepLabV3Plus", "DLV3P")
    # update_project_config(runs, "encoder_name", "timm-efficientnet-b4", "EfficientNet")
    # update_project_config(runs, "encoder_name", "timm-mobilenetv3_large_100", "MobileNet")
    # update_project_config(runs, "encoder_name", "resnet50", "ResNet")
    # update_project_config(runs, "encoder_name", "xception", "Xception")
