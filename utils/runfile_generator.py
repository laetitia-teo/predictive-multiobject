"""
Generates slurm runfiles for all configurations in the config folder.
"""
import argparse
import re
import os
import os.path as op

parser = argparse.ArgumentParser()

add = parser.add_argument

add('--min-cpus', type=int, default=12)
add('--expe-duration', type=int, default=20)
add('--min-config', type=int, default=0)

args = parser.parse_args()

# get all config paths
config_dir = '../configs'
paths = os.listdir(config_dir)

# def parse_fn(p):
#     s = re.search(r'config([0-9]+)', p)
#     if

c_inds = []
for p in paths:
    s = re.search(r'config([0-9]+)', p)
    if s:
        if int(s[1]) >= args.min_config:
            c_inds.append(s[1])


for c_idx in c_inds:

    experiment_name = f"config{c_idx}"
    runfile_name = op.join(
        "..",
        "runfiles",
        f"run{c_idx}.sh"
    )

    s = ("#!/bin/sh\n"
    f"#SBATCH --mincpus {args.min_cpus}\n"
    f"#SBATCH -t {args.expe_duration}:00:00\n"
    f"#SBATCH -o saves/{experiment_name}/log.out\n"
    f"#SBATCH -e saves/{experiment_name}/log.err\n"
    f"python main.py -c {c_idx} -n {experiment_name}\n"
    "wait\n")

    with open(runfile_name, 'w') as f:
        f.write(s)