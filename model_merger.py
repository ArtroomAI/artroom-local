import os
import argparse
import torch
import tqdm
import time
import math

try:
    # Made by Yuss#5555


    parser = argparse.ArgumentParser(description="Merge two models")
    parser.add_argument("model_0", type=str, help="Path to model 0")
    parser.add_argument("model_1", type=str, help="Path to model 1")
    parser.add_argument("--model_2", type=str, help="Path to model 2. IF THIS IS SET, --method will be ignored and 'add difference' will be used.",required=False,default=None)
    parser.add_argument("--alpha", type=float, help="Alpha value, optional, defaults to 0.5 (Overwritten by --fullrange)", default=0.5, required=False)
    parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
    parser.add_argument("--fullrange", action='store_true', help="generate merges for every 5 percent interval between 5 and 95", default=False, required=False)
    parser.add_argument("--method", type=str, help="Select interpolation method from 'sigmoid' 'inverse_sigmoid' 'weighted_sum'. defaults to 'weighted_sum'.", default="weighted_sum", required=False)
    parser.add_argument("--steps", type=int, help="Select interpolation steps at which the Models will be merged. 5 will result in 5% 10% 15% 20% .defaults to '10'.", default=10, required=False)

    args = parser.parse_args()

    # Copy paste from extras.py

    # Linear interpolation (https://en.wikipedia.org/wiki/Linear_interpolation)
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

    # Copied from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/fdecb636855748e03efc40c846a0043800aadfcc/modules/extras.py
    # Line 178 and 179
    def add_difference(theta0, theta1, theta2, alpha):
        return theta0 + (theta1 - theta2) * (1.0 - alpha)

    print(args)

    theta_funcs = {
        "weighted_sum": weighted_sum,
        "sigmoid": sigmoid,
        "inverse_sigmoid": inv_sigmoid,
    }

    models_path = args.model_0.rsplit('\\', 1)[0]
    print(models_path)

    modelName_0 = args.model_0.rsplit('\\', 1)[-1][:-5]
    modelName_1 = args.model_1.rsplit('\\', 1)[-1][:-5]


    names = ""


    def merge_models(model_0, model_1, alpha, output=None):
        '''consolidate merging models into a helpful function for the purpose of generating a range of merges'''
        model_0 = torch.load(model_0)
        model_1 = torch.load(model_1)
        theta_0 = model_0["state_dict"]
        theta_1 = model_1["state_dict"]
        alpha = alpha

        if not os.path.exists(f"{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}"):
            os.makedirs(f"{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}")

        theta_func = theta_funcs[args.method]

        if output is None:
            #output_file = f'merge-{modelName_0}-{math.ceil(alpha*100)}%-_WITH_-{modelName_1}-{math.ceil(100-alpha*100)}%.ckpt' #too long
            output_file = f'{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}/{modelName_0}-{round(alpha*100)}%--{modelName_1}-{round(100-(alpha*100))}%.ckpt'
        else:
            output_file = f'{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}/{output}-{str(alpha)[2:] + "0"}.ckpt'

        for key in tqdm.tqdm(theta_0.keys()):
            if 'model' in key and key in theta_1:
                theta_0[key] = theta_func(theta_0[key], theta_1[key], (float(1.0) - alpha))

        for key in theta_1.keys():
            if 'model' in key and key not in theta_0:
                theta_0[key] = theta_1[key]



        print(f"Saving as {output_file}\n")

        torch.save(model_0, output_file)
        return output_file.rsplit("/", 1)[-1]

    def merge_three(model_0, model_1, alpha, output=None):
        '''consolidate merging models into a helpful function for the purpose of generating a range of merges'''
        model_0 = torch.load(model_0)
        model_1 = torch.load(model_1)
        model_2 = torch.load(args.model_2)
        theta_0 = model_0["state_dict"]
        theta_1 = model_1["state_dict"]
        theta_2 = model_2["state_dict"]
        alpha = alpha


        theta_func = add_difference

        if not os.path.exists(f"{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}"):
            os.makedirs(f"{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}")

        if output is None:
            #output_file = f'merge-{modelName_0}-{math.ceil(alpha*100)}%-_WITH_-{modelName_1}-{math.ceil(100-alpha*100)}%.ckpt' #too long
            output_file = f'{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}/{modelName_0}-{round(alpha*100)}-{modelName_1}-{round(100-alpha*100)}_3_{args.model_2.rsplit("/", 1)[-1][:-5]}.ckpt'
        else:
            output_file = f'{models_path}/merge-{modelName_0}_{modelName_1}-{args.method}/{output}-{str(alpha)[2:] + "0"}.ckpt'

        for key in tqdm.tqdm(theta_0.keys()):
            if 'model' in key and key in theta_1:
                t2 = (theta_2 or {}).get(key)
                if t2 is None:
                    t2 = torch.zeros_like(theta_0[key])
                theta_0[key] = theta_func(theta_0[key], theta_1[key], t2, (float(1.0) - alpha))  # Need to reverse the interp_amount to match the desired mix ration in the merged checkpoint

    


        for key in theta_1.keys():
            if 'model' in key and key not in theta_0:
                theta_0[key] = theta_1[key]



        print(f"Saving as {output_file}\n")

        torch.save(model_0, output_file)
        return output_file.rsplit("/", 1)[-1]




    if args.fullrange and args.steps != 0:
        for i in range(args.steps, 100, args.steps):
            print(f"Merging {modelName_0} with {modelName_1} at {i}% interpolation with {args.method}")
            if(args.model_2 is not None): 
                names += merge_three(args.model_0, args.model_1, i/100) + ","
            else:
                names += merge_models(args.model_0, args.model_1, i/100) + ","

            # Probably not an issue. Remove if not needed:
            time.sleep(0.5) # make extra sure the gc has some extra time to dump torch stuff just in case ??
        print(names)
    else:
        print(f"Merging {modelName_0} with {modelName_1} at {args.alpha}% interpolation with {args.method}")
        names += merge_models(args.model_0, args.model_1, args.alpha / 100., args.output) + ","

except:
    print("Something went wrong. Share the console output. We might fix it")
    while True:
        None
