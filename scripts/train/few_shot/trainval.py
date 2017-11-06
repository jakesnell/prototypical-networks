import os
import json
import subprocess

from protonets.utils import format_opts, merge_dict
from protonets.utils.log import load_trace

def main(opt):
    result_dir = os.path.dirname(opt['model.model_path'])

    # get target training loss to exceed
    trace_file = os.path.join(result_dir, 'trace.txt')
    trace_vals = load_trace(trace_file)
    best_epoch = trace_vals['val']['loss'].argmin()

    # load opts
    model_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # override previous training ops
    model_opt = merge_dict(model_opt, {
        'log.exp_dir': os.path.join(model_opt['log.exp_dir'], 'trainval'),
        'data.trainval': True,
        'train.epochs': best_epoch + model_opt['train.patience'],
    })

    subprocess.call(['python', os.path.join(os.getcwd(), 'scripts/train/few_shot/run_train.py')] + format_opts(model_opt))
