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

# constants 
LAVIS_ROOT = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT'
CACHE_ROOT = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/export/home/.cache/lavis'
METRIC_ROOT = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/report-generation-baseline/CXR-Report-Metric'
# DATE = '0601'

os.chdir(LAVIS_ROOT)
from lavis.common.registry import registry


parser = argparse.ArgumentParser()
parser.add_argument('--eval_cfg_path', type=str, default='lavis/projects/blip2/albef_vit_0531/0607_ft_vicuna_findings_wval_eval.yaml')
parser.add_argument('--date', type=str, default='')
# parser.add_argument('--job_id', type=str, default='20230311232')
parser.add_argument('--result_file', type=str, 
                    # default='/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian/lavis/output/BLIP2/Finetune/from_pt2_impression_full/20230311200/result/test_epochbest.json',
                    default='')
parser.add_argument('--eval_mode', type=str, default='findings')
parser.add_argument('--eval_level', type=str, default='image')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--eval_split', type=str, default='test') # rarely change


## Run Metrics
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


eval_mode = 'findings'
eval_level = 'image'
result_file = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT/lavis/output/BLIP2/Finetune/Vicuna/Findings_wval_subset/20230608023/ckpt74/20230608044/result/test_epochbest.json'

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
    print("gen file", gen_file)
    subprocess.run(f'cp {result_file} {gen_file}', shell=True)
    
    ### convert json to csv
    csv_file = convert_to_csv(gen_file, gt_file)

    ### copy this csv file to the metrics/test_results folder
    dest_csv_file_base=run_name
    csv_dest_file = os.path.join(METRIC_ROOT, 'test_results', dest_csv_file_base)
    subprocess.run(f'cp {csv_file} {csv_dest_file}', shell=True)

    cmd_run_metrics = f"source ~/.bashrc; conda activate cxr_metrics; cd {METRIC_ROOT}; python test_metric_csv.py --res_csv={dest_csv_file_base} --field='{eval_mode}' --eval_level='{eval_level}'"

    subprocess.run(cmd_run_metrics, shell=True)
    
def main(args):
    checkpoint_path_arg = args.checkpoint_path if args.checkpoint_path is not '' else None
    run_metrics(eval_cfg_path=args.eval_cfg_path, 
                eval_mode=args.eval_mode, 
                eval_level=args.eval_level, 
                result_file=args.result_file,
                checkpoint_path=checkpoint_path_arg,
                eval_split=args.eval_split)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)