import os
import argparse
import torch
from tqdm import tqdm
import time
import math
from safe import load as safe_load
from safetensors import safe_open
from safetensors.torch import save_file
import ctypes

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

def add_difference(theta0, theta1, theta2, alpha):
    return theta0 + (theta1 - theta2) * (1.0 - alpha)

def load_model_from_config(ckpt, use_safe_load=True):
    print(f"Loading model from {ckpt}")
    if ".safetensors" in ckpt:
        pl_sd = {}
        with safe_open(ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                pl_sd[key] = f.get_tensor(key)
    elif use_safe_load:
        pl_sd = safe_load(ckpt)
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")

    if "state_dict" in pl_sd:
        return pl_sd["state_dict"]
    return pl_sd

class ModelMerger:
    def __init__(self, data):
        print(data)
        self.data = data
        self.output_path = os.path.dirname(data["model_0"])
        self.output_ext = os.path.splitext(data["model_0"])[1]

        self.modelName_0 = os.path.basename(data["model_0"]).split('.')[0]
        self.modelName_1 = os.path.basename(data["model_1"]).split('.')[0]
        self.modelName_2 = ""

        self.model_0 = load_model_from_config(data["model_0"])
        self.model_1 = load_model_from_config(data["model_1"])
        self.model_2 = None

        if data["model_2"]:
            self.modelName_2 = os.path.basename(data["model_2"]).split('.')[0]
            self.model_2 = load_model_from_config(data["model_2"])

    def save_file(self, new_model, output_file):
        print(f"Saving as {output_file}\n")

        if self.output_ext == 'safetensors':
            save_file(new_model, output_file)
        else:
            torch.save(new_model, output_file)
        return output_file.rsplit("/", 1)[-1]

    def merge_models(self, alpha):
        """Consolidate merging models into a helpful function for the purpose of generating a range of merges"""

        output_dir = f"{self.output_path}/merge-{self.modelName_0}_{self.modelName_1}-{self.data['method']}"

        os.makedirs(output_dir, exist_ok=True)

        theta_funcs = {
            "weighted_sum": weighted_sum,
            "sigmoid": sigmoid,
            "inverse_sigmoid": inv_sigmoid,
        }
        theta_func = theta_funcs[self.data['method']]

        new_model = dict()

        if self.data['output'] == '':
            output_file = f'{output_dir}/{self.modelName_0}-{round(alpha * 100)}%--{self.modelName_1}-{round(100 - alpha * 100)}%{self.output_ext}'
        else:
            output_file = f"{output_dir}/{self.data['output']}-{round(alpha * 100)}%{self.output_ext}"

        for key in tqdm(self.model_0.keys()):
            if 'model' in key and key in self.model_1:
                new_model[key] = theta_func(self.model_0[key], self.model_1[key], (float(1.0) - alpha))

        for key in self.model_1.keys():
            if 'model' in key and key not in self.model_0:
                new_model[key] = self.model_1[key]

        return self.save_file(new_model, output_file)

    def merge_three(self, alpha):
        """consolidate merging models into a helpful function for the purpose of generating a range of merges"""
        theta_func = add_difference

        output_dir = f"{self.output_path}/merge-{self.modelName_0}_{self.modelName_1}-{self.data['method']}"

        new_model = dict()

        os.makedirs(output_dir, exist_ok=True)

        if self.data['output'] == '':
            output_file = f'{output_dir}/{self.modelName_0}-{round(alpha * 100)}-{self.modelName_1}-{round(100 - alpha * 100)}_3_{self.modelName_2}{self.output_ext}'
        else:
            output_file = f"{output_dir}/{self.data['output']}-{round(alpha * 100)}{self.output_ext}"

        for key in tqdm(self.model_0.keys()):
            if 'model' in key and key in self.model_1:
                t2 = (self.model_2 or {}).get(key)
                if t2 is None:
                    t2 = torch.zeros_like(self.model_0[key])
                new_model[key] = theta_func(self.model_0[key], self.model_1[key], t2, (float(
                    1.0) - alpha))  # Need to reverse the interp_amount to match the desired mix ration in the merged checkpoint

        for key in self.model_1.keys():
            if 'model' in key and key not in self.model_0:
                new_model[key] = self.model_1[key]

        return self.save_file(new_model, output_file)

    def run(self):
        names = ""
        if self.data["steps"] != 0:
            print("Running with incremental steps")
            print(f"Start: {self.data['start_steps']} End: {self.data['end_steps']} Increment: {self.data['steps']}")
            time.sleep(5)
            for i in range(self.data['start_steps'], self.data['end_steps'], self.data['steps']):
                print(f"Merging {self.modelName_0} with {self.modelName_1} at {i}% interpolation with {self.data['method']}")
                if self.data['model_2'] != "":
                    names += self.merge_three(i / 100.) + ","
                else:
                    names += self.merge_models(i / 100.) + ","

                # Probably not an issue. Remove if not needed:
                time.sleep(0.5)  # make extra sure the gc has some extra time to dump torch stuff just in case ??
        else:
            print(f"Merging {self.modelName_0} with {self.modelName_1} at {self.data['alpha']}% interpolation with {self.data['method']}")
            if self.data['model_2'] != "":
                names = self.merge_three(self.data['alpha'] / 100.)
            else:
                names = self.merge_models(self.data['alpha'] / 100.)
        print(names)
