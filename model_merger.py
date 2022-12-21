import os
import argparse
import torch
from tqdm import tqdm
import time
import math
from safe import load as safe_load
import ctypes

"""
@Credit: Yuss#5555
"""

# Prevents console from freezing due to Windows being dumb
kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)


def weighted_sum(theta0, theta1, alpha):
    return ((1 - alpha) * theta0) + (alpha * theta1)


# Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
def sigmoid(theta0, theta1, alpha):
    alpha = alpha * alpha * (3 - (2 * alpha))
    return theta0 + ((theta1 - theta0) * alpha)


# Inverse Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
def inv_sigmoid(theta0, theta1, alpha):
    alpha = 0.5 - math.sin(math.asin(1.0 - 2.0 * alpha) / 3.0)
    return theta0 + ((theta1 - theta0) * alpha)


# Line 178 and 179
def add_difference(theta0, theta1, theta2, alpha):
    return theta0 + (theta1 - theta2) * (1.0 - alpha)


def load_model_from_config(ckpt):
    pl_sd = safe_load(ckpt)
    if "state_dict" in pl_sd:
        return pl_sd["state_dict"]
    return pl_sd


def merge_models(model_0, model_1, alpha, output=None):
    """Consolidate merging models into a helpful function for the purpose of generating a range of merges"""

    model_0 = load_model_from_config(model_0)
    model_1 = load_model_from_config(model_1)

    if not os.path.exists(f"{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}"):
        os.makedirs(f"{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}")

    theta_func = theta_funcs[args.method]

    if output is None:
        output_file = f'{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}/{modelName_0}-{round(alpha * 100)}%--{modelName_1}-{round(100 - alpha * 100)}%.ckpt'
    else:
        output_file = f'{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}/{output}-{round(alpha * 100)}%.{model_ext_0}'

    for key in tqdm(model_0.keys()):
        if 'model' in key and key in model_1:
            model_0[key] = theta_func(model_0[key], model_1[key], (float(1.0) - alpha))

    for key in model_1.keys():
        if 'model' in key and key not in model_0:
            model_0[key] = model_1[key]

    print(f"Saving as {output_file}\n")

    torch.save(model_0, output_file)
    return output_file.rsplit("/", 1)[-1]


def merge_three(model_0, model_1, alpha, output=None):
    """consolidate merging models into a helpful function for the purpose of generating a range of merges"""
    model_0 = load_model_from_config(model_0)
    model_1 = load_model_from_config(model_1)
    model_2 = load_model_from_config(args.model_2)

    theta_func = add_difference

    if not os.path.exists(f"{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}"):
        os.makedirs(f"{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}")

    if output is None:
        output_file = f'{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}/{modelName_0}-{round(alpha * 100)}-{modelName_1}-{round(100 - alpha * 100)}_3_{modelName_2}.{model_ext_0}'
    else:
        output_file = f'{models_path}//merge-{modelName_0}_{modelName_1}-{args.method}/{output}-{round(alpha * 100)}%.{model_ext_0}'

    for key in tqdm(model_0.keys()):
        if 'model' in key and key in model_1:
            t2 = (model_2 or {}).get(key)
            if t2 is None:
                t2 = torch.zeros_like(model_0[key])
            model_0[key] = theta_func(model_0[key], model_1[key], t2, (float(
                1.0) - alpha))  # Need to reverse the interp_amount to match the desired mix ration in the merged checkpoint

    for key in model_1.keys():
        if 'model' in key and key not in model_0:
            model_0[key] = model_1[key]

    print(f"Saving as {output_file}\n")

    torch.save(model_0, output_file)
    return output_file.rsplit("/", 1)[-1]


try:
    parser = argparse.ArgumentParser(description="Merge two models")
    parser.add_argument("model_0", type=str, help="Path to model 0")
    parser.add_argument("model_1", type=str, help="Path to model 1")
    parser.add_argument("--model_2", type=str,
                        help="Path to model 2. IF THIS IS SET, --method will be ignored and 'add difference' will be used.",
                        required=False, default=None)
    parser.add_argument("--alpha", type=float, help="Alpha value, optional, defaults to 0.5", default=0.5,
                        required=False)
    parser.add_argument("--output", type=str, help="Output file name, without extension", required=False)
    parser.add_argument("--method", type=str,
                        help="Select interpolation method from 'sigmoid' 'inverse_sigmoid' 'weighted_sum'. defaults to 'weighted_sum'.",
                        default="weighted_sum", required=False)
    parser.add_argument("--steps", type=int,
                        help="Select interpolation steps at which the Models will be merged. 5 will result in 5% 10% 15% 20% .defaults to '10'.",
                        default=10, required=False)
    parser.add_argument("--start_steps", type=int, help="Where to start the steps, default 0", default=0,
                        required=False)
    parser.add_argument("--end_steps", type=int, help="Where to end the steps, default 100", default=100,
                        required=False)
    args = parser.parse_args()

    print(args)

    theta_funcs = {
        "weighted_sum": weighted_sum,
        "sigmoid": sigmoid,
        "inverse_sigmoid": inv_sigmoid,
    }

    models_path = os.path.split(args.model_0)[0]
    print(models_path)

    # Weird but handles cases when there is a . in the name
    modelName_0 = os.path.basename(args.model_0).split('.')[0]
    model_ext_0 = os.path.basename(args.model_0).split('.')[-1]
    modelName_1 = os.path.basename(args.model_1).split('.')[0]
    model_ext_1 = os.path.basename(args.model_1).split('.')[-1]
    if args.model_2:
        modelName_2 = os.path.basename(args.model_2).split('.')[0]
        model_ext_2 = os.path.basename(args.model_2).split('.')[-1]
    names = ""

    if args.steps != 0:
        for i in range(args.start_steps, args.end_steps, args.steps):
            print(f"Merging {modelName_0} with {modelName_1} at {i}% interpolation with {args.method}")
            if args.model_2 is not None:
                names += merge_three(args.model_0, args.model_1, i / 100) + ","
            else:
                names += merge_models(args.model_0, args.model_1, i / 100) + ","

            # Probably not an issue. Remove if not needed:
            time.sleep(0.5)  # make extra sure the gc has some extra time to dump torch stuff just in case ??
        print(names)
    else:
        print(f"Merging {modelName_0} with {modelName_1} at {args.alpha}% interpolation with {args.method}")
        names += merge_models(args.model_0, args.model_1, args.alpha / 100., args.output) + ","

except Exception as e:
    print(e)
    print("Something went wrong. Share the console output. We will fix it")
    time.sleep(300)
