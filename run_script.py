import os 
import pandas as pd 
import json 
import yaml 
import shutil
import csv 
import sys 
from omegaconf import OmegaConf
import subprocess
import argparse
import random 
import numpy as np

# constants 
LAVIS_ROOT = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT'
CACHE_ROOT = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/export/home/.cache/lavis'
METRIC_ROOT = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/report-generation-baseline/CXR-Report-Metric'
# DATE = '0601'

os.chdir(LAVIS_ROOT)
print("PWD: ", os.getcwd())
from lavis.common.registry import registry


parser = argparse.ArgumentParser()
parser.add_argument('--run_train', action='store_true')
parser.add_argument('--create_eval_config', action='store_true')
parser.add_argument('--run_evaluate', action='store_true')
parser.add_argument('--run_metrics', action='store_true')


parser.add_argument('--train_cfg_path', type=str, default='')
parser.add_argument('--eval_cfg_path', type=str, default='') # lavis/projects/blip2/albef_vit_0531/0607_ft_vicuna_findings_wval_eval.yaml
parser.add_argument('--date', type=str, default='')
# parser.add_argument('--job_id', type=str, default='20230311232')
parser.add_argument('--result_file', type=str, 
                    # default='/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian/lavis/output/BLIP2/Finetune/from_pt2_impression_full/20230311200/result/test_epochbest.json',
                    default='')
parser.add_argument('--eval_mode', type=str, default='findings')
parser.add_argument('--eval_level', type=str, default='image')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--eval_split', type=str, default='test') # rarely change

train_config_path = 'lavis/projects/blip2/albef_vit_0531/0609_ft_vicuna_find_wval_ins2.yaml'
checkpoint_path = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT/lavis/output/BLIP2/Finetune/Vicuna/Findings_wval_subset_ins2/20230610012/checkpoint_48.pth'


### get temp log file name ### 
def get_log_file(base):
    i = random.randint(0, 100000)
    log_file = f'_{base}_log{i}.txt'
    while os.path.exists(f'_{base}_log{i}.txt'):
        i += 1
        log_file = f'_{base}_log{i}.txt'
    return log_file


#### Run Train ####
def run_train(train_config_path):
    assert train_config_path.endswith('.yaml'), f"Train config ({train_config_path}) path must be a yaml file"
    cmd_train = f'python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path {train_config_path}'
    print("Cmd Train: ", cmd_train)
    log_file = get_log_file('train')
    print("Log File: ", log_file)
    cmd_train_log = f'{cmd_train} | tee {log_file}'
    subprocess.run(cmd_train_log, shell=True)

    # read first line of log file
    with open(log_file, 'r') as f:
        first_line = f.readline()
        print("Log First Line: ", first_line)
    
    # get job id
    assert first_line.startswith('FOLDER NAME'), f"Job Id not found in first line of log file"
    job_id = first_line.split(' ')[-1].strip()
    return job_id
    

#### Create Eval Config #### 
def get_eval_config_path(train_config_path):
    return train_config_path.split('.')[0] + '_eval.yaml'

def create_eval_config(train_config_path, checkpoint_path, eval_split):
    assert train_config_path.endswith('.yaml'), f"Train config ({train_config_path}) path must be a yaml file"
    assert checkpoint_path.endswith('.pth'), f"Checkpoint path ({checkpoint_path}) must be a pth file"

    train_config_path_full = os.path.join(LAVIS_ROOT, train_config_path)

    # here is where we will save the evaluate config 
    eval_config_path = get_eval_config_path(train_config_path)
    print("Eval Config Path: ", eval_config_path)
    eval_config_path_full = os.path.join(LAVIS_ROOT, eval_config_path)

    # extract the checkpoint number 
    checkpoint_num = checkpoint_path.split('/')[-1].split('.')[0].split('_')[-1]
    print("Checkpoint Num: ", checkpoint_num)

    # construct the output dir 
    eval_output_dir = '/'.join(checkpoint_path[len(LAVIS_ROOT)+7:].split('/')[:-1] + [f'ckpt{checkpoint_num}'])
    eval_output_dir += '/'
    print("Eval Output Dir: ", eval_output_dir)

    train_yaml = OmegaConf.load(train_config_path_full)

    # construct and save the evaluate config 
    eval_yaml = train_yaml.copy()
    eval_yaml['model']['finetuned'] = checkpoint_path 
    eval_yaml['model']['load_finetuned'] = True 
    eval_yaml['run']['evaluate'] = True 
    eval_yaml['run']['output_dir'] = eval_output_dir
    eval_yaml['run']['batch_size_eval'] = eval_yaml['run']['batch_size_train']
    del eval_yaml['run']['train_splits']
    del eval_yaml['run']['valid_splits']
    eval_yaml['run']['test_splits'] = [eval_split]

    OmegaConf.save(eval_yaml, eval_config_path_full)
    print("Saved eval config to Eval Config Path.")

    return eval_config_path


#### Run evaluate.py #### 
def run_evaluate(eval_config_path):
    assert eval_config_path.endswith('.yaml'), f"Evaluate config ({eval_config_path}) path must be a yaml file"
    conda_env_name = 'lvs_vit' # TODO: can make this a parameter
    
    subprocess.run(f'cd {LAVIS_ROOT}', shell=True)
    subprocess.run(f'source ~/.bashrc; conda activate {conda_env_name}', shell=True)

    cmd_evaluate = f'python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path {eval_config_path}'
    print("Cmd Evaluate: ", cmd_evaluate)

    log_file = get_log_file('eval')
    print("Log File: ", log_file)
    cmd_evaluate_log = f'{cmd_evaluate} | tee {log_file}'
    subprocess.run(cmd_evaluate_log, shell=True)

    # read log
    with open(log_file, 'r') as f:
        lines = f.readlines()
    j = len(lines) - 1 
    while j >= 0:
        line = lines[j]
        if line.startswith('result file saved to '):
            # extract the result file
            result_file = line.split('result file saved to ')[-1].strip()
            print("Result File: ", result_file)
            break
        j -= 1
    assert j >= 0, "Could not find result file in log file"
    # delete log file
    subprocess.run(f'rm {log_file}', shell=True)
    # return result file
    return result_file


#### Run Metrics ####
def convert_to_csv(gen_file, gt_file):
    '''
    convert the result file to csv 
    '''
    ### save the csv to same directory as the gen_file, just with new extension
    csv_file = f'{gen_file[:-4]}csv'

    def load_from_json(json_file, descr=""):
        f = open(json_file, 'r')
        json_data = json.load(f)
        f.close()
        print(f"Loaded {descr} from {json_file}. Len = {len(json_data)}.")
        return json_data
    
    gen_data = load_from_json(gen_file, descr="generations")
    gt_data = load_from_json(gt_file, descr="ground truth")
    print("Example gen item:")
    print(gen_data[0])
    print("Example gt item:")
    print(gt_data[0])

    def get_from_item(item, report_mode = 'findings'):
        if report_mode in item:
            return item[report_mode]
        elif 'caption' in item:
            return item['caption']
        else:
            return ""
    
    ### setting up csv data 
    header = ['subject_id', 'study_id', 'dicom_id', 'findings', 'impression', 'target']
    csv_data = []
    csv_data.append(header)
    # TODO: maybe need to sort by image_id, but seems already sorted. 
    # assume gen_data and gt_data are the same len and sorted by image_id
    for gen_item, gt_item in zip(gen_data, gt_data):

        subject_id = gt_item['subject_id'][1:]
        study_id = gt_item['study_id'][1:]
        dicom_id = gt_item['dicom_id']

        # TODO: don't need to save 3 times right now since i am using 'caption'
        findings = get_from_item(gen_item, 'findings').strip("\"\'")
        impression = get_from_item(gen_item, 'impression').strip("\"\'")
        target = get_from_item(gen_item, 'target').strip("\"\'")

        # print(f"FINDINGS: {findings} END")
        # print(f"IMPRESSION: {impression} END")

        row = [subject_id, study_id, dicom_id, findings, impression, target]
        csv_data.append(row)
    print("Finished creating csv data.")

    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
        print(f"Saved to {csv_file}")
    return csv_file 


def run_metrics(eval_config_path, eval_mode, eval_level, result_file, checkpoint_path=None, eval_split='test'):
    '''
    eval_split (the split we are evaluating results on, usually test)
    '''
    # eval_config_path = 'lavis/projects/blip2/albef_vit_0531/0607_ft_vicuna_findings_wval_eval.yaml'
    eval_config_path_full = os.path.join(LAVIS_ROOT, eval_config_path)
    eval_config = OmegaConf.load(eval_config_path_full)

    ### get checkpoint path and checkpoint num
    if checkpoint_path is None:
        checkpoint_path = eval_config['model']['finetuned']
    checkpoint_num = checkpoint_path.split('/')[-1].split('.')[0].split('_')[-1]
    print("Checkpoint Num: ", checkpoint_num)

    ### get run name
    run_name = eval_config_path.split('/')[-1].split('.')[0]
    if run_name[-5:] == '_eval':
        run_name = run_name[:-5]
    print("Run Name: ", run_name)

    ### set up experiment directory 
    exper_dir = os.path.join(LAVIS_ROOT, 'kat_dev', 'experiments', run_name)
    print("Experiment Dir: ", exper_dir)
    if not os.path.exists(exper_dir):
        os.makedirs(exper_dir)

    ### extract gt file automatically!
    ### get dataset name, assuming list size = 1
    dataset_name = list(eval_config.get('datasets').keys())[0]
    print("Dataset Name: ", dataset_name)
    builder_cls = registry.get_builder_class(dataset_name)
    dataset_config_type = 'default'
    # dataset_config_type = datasets[dataset_name].get("type", "default")
    dataset_config_path = builder_cls.default_config_path(type=dataset_config_type)
    dataset_info = OmegaConf.load(dataset_config_path)
    dataset_info['datasets'][dataset_name]['build_info']['annotations'][eval_split]['storage']
    gt_file = os.path.join(CACHE_ROOT, dataset_info['datasets'][dataset_name]['build_info']['annotations'][eval_split]['storage'])
    print("GT File: ", gt_file)
    
    ### convert the result_file to a file with the number (this step could be skipped later if we fix evaluate.py)
    ### num_filename: .../test_epoch3.json
    num_filename = result_file.split('/')[-1].split('.')[0][:-len("best")] + f'{checkpoint_num}.json'
    gen_file = os.path.join(exper_dir, num_filename)
    print("GEN File", gen_file)
    subprocess.run(f'cp {result_file} {gen_file}', shell=True)
    
    ### convert json to csv
    csv_file = convert_to_csv(gen_file, gt_file)

    ### copy this csv file to the metrics/test_results folder
    dest_csv_file_base=run_name
    csv_dest_file = os.path.join(METRIC_ROOT, 'test_results', dest_csv_file_base)
    subprocess.run(f'cp {csv_file} {csv_dest_file}', shell=True)

    cmd_run_metrics = f"source ~/.bashrc; conda activate cxr_metrics; cd {METRIC_ROOT}; python test_metric_csv.py --res_csv={dest_csv_file_base} --field='{eval_mode}' --eval_level='{eval_level}'"
    print("Cmd Run Metrics: ", cmd_run_metrics)
    subprocess.run(cmd_run_metrics, shell=True)


def main(args):
    if args.eval_cfg_path:
        eval_config_path_arg = args.eval_cfg_path
    elif args.train_cfg_path:
        eval_config_path_arg = get_eval_config_path(args.train_cfg_path)
    result_file_arg = args.result_file 

    ### start run ### 
    if args.run_train:
        print("\nRUN TRAIN")
        job_id = run_train(args.train_cfg_path)

    if args.create_eval_config:
        print("\nCREATE EVAL CONFIG")
        eval_config_path_arg = create_eval_config(args.train_cfg_path, args.checkpoint_path, args.eval_split)
    
    if args.run_evaluate:
        print("\nRUN EVALUATE")
        result_file_arg = run_evaluate(eval_config_path_arg)
    
    if args.run_metrics:
        print("\nRUN METRICS")
        checkpoint_path_arg = None if args.checkpoint_path=='' else args.checkpoint_path
        run_metrics(eval_config_path=eval_config_path_arg, 
                    eval_mode=args.eval_mode, 
                    eval_level=args.eval_level, 
                    result_file=result_file_arg,
                    checkpoint_path=checkpoint_path_arg,
                    eval_split=args.eval_split)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)