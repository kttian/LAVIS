import os 
import pandas as pd 
import json 
import yaml 
import shutil
import csv 
import sys 
from omegaconf import OmegaConf
import subprocess



# constants 
LAVIS_ROOT = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT'
CACHE_ROOT = "/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/export/home/.cache/lavis"
DATE = '0601'
RUN_NAME = 'vicuna_target_wval'

os.chdir(LAVIS_ROOT)
from lavis.common.registry import registry

### 
exper_dir = os.path.join(LAVIS_ROOT, 'kat_dev', 'experiments', RUN_NAME)
print("Experiment Dir: ", exper_dir)
if not os.path.exists(exper_dir):
    os.makedirs(exper_dir)

## Step 1: Run evaluate.py, given a train job and checkpoint path
## Construct the evaluate config 

# given 
# train_config_path = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT/lavis/projects/blip2/train/llama_llm/ft_vicuna_target_wval.yaml'
train_config_path = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT/kat_dev/experiments/vicuna_target_wval/ft_vicuna_target_wval.yaml'
# checkpoint_path = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT/lavis/output/BLIP2/Finetune/Vicuna/Target_wval/20230530165/checkpoint_10.pth'

checkpoint_path = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT/lavis/output/BLIP2/Finetune/ALBEF_PTOPT/Findings_wval/20230530164/checkpoint_29.pth'

# # here is where we will save the evaluate config 
# eval_config_path = train_config_path.split('.')[0] + '_eval.yaml'
# print("Eval Config Path: ", eval_config_path)

eval_config_path = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT/lavis/projects/blip2/eval/albef_vit/ft_find_wval_pre2b_eval.yaml '

# # extract the checkpoint number 
checkpoint_num = checkpoint_path.split('/')[-1].split('.')[0].split('_')[-1]
# print("Checkpoint Num: ", checkpoint_num)

# # construct the output dir 
# eval_output_dir = '/'.join(checkpoint_path[len(LAVIS_ROOT)+7:].split('/')[:-1] + [f'ckpt{checkpoint_num}'])
# eval_output_dir += '/'
# print("Eval Output Dir: ", eval_output_dir)

# # open the train config 
# with open(os.path.join(LAVIS_ROOT, train_config_path), 'r') as file:
#     train_yaml = yaml.safe_load(file)
# # print(train_yaml)

# # construct and save the evaluate config 
# eval_yaml = train_yaml.copy()
# eval_yaml['model']['finetuned'] = checkpoint_path 
# eval_yaml['run']['evaluate'] = True 
# eval_yaml['run']['output_dir'] = eval_output_dir
# eval_yaml['run']['batch_size_eval'] = eval_yaml['run']['batch_size_train']
# del eval_yaml['run']['train_splits']
# del eval_yaml['run']['valid_splits']
# with open(os.path.join(LAVIS_ROOT, eval_config_path), 'w') as file:
#     yaml.dump(eval_yaml, file, default_flow_style=False)


# sys.exit(1)
# ## Run Evaluate 
# # create eval sbatch file 
# eval_sbatch_filename = RUN_NAME + '_eval.sh'
# eval_sbatch_path = os.path.join(exper_dir, eval_sbatch_filename)
# print("Eval Sbatch Path: ", eval_sbatch_path)

# NUM_GPUS = 1
# mem_per_gpu = {1:50, 2:64, 4:84}
# time_per_gpu = {1:100, 2:80, 4:38}

# f = open(eval_sbatch_path, "w")

# f.write(f"#!/bin/bash\n\n")
# f.write(f"#SBATCH --partition gpu_quad\n")
# f.write(f"#SBATCH --mail-type=FAIL,RUNNING,COMPLETE\n")
# f.write(f"#SBATCH --gres=gpu:{NUM_GPUS},vram:48G\n")
# f.write(f"#SBATCH --mem={mem_per_gpu[NUM_GPUS]}G\n")
# f.write(f"#SBATCH --time={time_per_gpu[NUM_GPUS]}:00:00\n")
# f.write(f"#SBATCH -o {RUN_NAME}_%j_run.out\n")
# f.write(f"#SBATCH -e {RUN_NAME}_%j_err.out\n\n")

# f.write(f"source ~/.bashrc\n")
# f.write(f"module load java/jdk-11.0.11\n")
# f.write(f"export MASTER_ADDR='127.0.0.1'\n\n")
# f.write(f"conda activate lvs_vit\n")
# f.write(f"cd $LAVISK\n")
# f.write(f"cd ../LAVIS-kttian-VIT\n\n")

# f.write(f"nvidia-smi\n\n")

# f.write(f"cat {eval_config_path}\n")
# f.write(f"export MASTER_PORT=54114\n")
# f.write(f"python -m torch.distributed.run --nproc_per_node={NUM_GPUS} --master_port=54114 evaluate.py --cfg-path {eval_config_path}\n")
# f.close()

# # run sbatch 
# os.system(f'sbatch {eval_sbatch_path}')

# eval_cmd = f'python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path {eval_config_path}'
# print(eval_cmd)
# os.system(f'cd {LAVIS_ROOT}')
# os.system(eval_cmd)


## Run Metrics
# given 

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

def run_metrics(eval_mode, eval_level, result_file, checkpoint_path=None, eval_split='test'):
    '''
    eval_split (the split we are evaluating results on, usually test)
    '''
    eval_config_path = 'lavis/projects/blip2/albef_vit_0531/0607_ft_vicuna_findings_wval_eval.yaml'
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
    os.system(f'cp {result_file} {gen_file}')
    
    ### convert json to csv
    csv_file = convert_to_csv(gen_file, gt_file)
    csv_file.split('/')[-1]

    ### copy this csv file to the metrics/test_results folder
    dest_csv_file_base=run_name
    csv_dest_file = os.path.join('/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/report-generation-baseline/CXR-Report-Metric/test_results/',dest_csv_file_base)
    os.system(f'cp {csv_file} {csv_dest_file}')

    cmd_run_metrics = f"source ~/.bashrc; conda activate cxr_metrics; cd /n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/report-generation-baseline/CXR-Report-Metric; python test_metric_csv.py --res_csv={dest_csv_file_base} --field='{eval_mode}' --eval_level='{eval_level}'"

    subprocess.run(cmd5, shell=True)
    
eval_mode = 'findings'
eval_level = 'image'

# result file from running evaluate.py: .../test_epochbest.json 
# result_file = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT/lavis/output/BLIP2/Finetune/Debug/20230601011/ckpt3/20230601012/result/test_epochbest.json'
result_file = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT/lavis/output/BLIP2/Finetune/ALBEF_PTOPT/Findings_wval/20230530164/ckpt29/20230607180/result/test_epochbest.json'
# num_filename: .../test_epoch3.json
num_filename = result_file.split('/')[-1].split('.')[0][:-len("best")] + f'{checkpoint_num}.json'
print(num_filename)

# gen_file: .../test_epoch3.json
gen_file = os.path.join(exper_dir, num_filename)
print("gen file", gen_file)
os.system(f'cp {result_file} {gen_file}')

# gt_file
# mimic_cxr_instruct_impression_noval_subset_224
# lavis/configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_impression_noval_subset_224.yaml
# mimic_cxr_0529/annotations/mimic_cxr_instruct_impression_noval_subset_test.json
# TODO: CHANGE
gt_file = os.path.join(CACHE_ROOT, f"mimic_cxr/annotations/mimic_test_findings_full_wval.json")
# gt_file = os.path.join(CACHE_ROOT, f"mimic_cxr_0529/annotations/mimic_cxr_instruct_{eval_mode}_wval_full_test.json")   

csv_file = os.path.join(exper_dir, f'{num_filename[:-4]}csv')

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

# %%
# setting up csv data 
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
    # print(impression)

    # print(f"FINDINGS: {findings} END")
    # print(f"IMPRESSION: {impression} END")

    row = [subject_id, study_id, dicom_id, findings, target]
    csv_data.append(row)
print("Finished creating csv data.")

# %%
with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
    print(f"Saved to {csv_file}")

# %%
csv_file.split('/')[-1]

# %%
dest_csv_file_base = f'{DATE}_{eval_mode}_wval.csv'
dest_csv_file_base

# %%
csv_dest_file = os.path.join('/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/report-generation-baseline/CXR-Report-Metric/test_results/',dest_csv_file_base)

# %%
csv_dest_file

# %%
os.system(f'cp {csv_file} {csv_dest_file}')

# %%
cmd5 = f"source ~/.bashrc; conda activate cxr_metrics; cd /n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/report-generation-baseline/CXR-Report-Metric; python test_metric_csv.py --res_csv={dest_csv_file_base} --field='{eval_mode}' --eval_level='{eval_level}'"

# %%
print(cmd5)

# %%
os.system(cmd5)

# %%
job_id = checkpoint_path.split('/')[-2]
date = '0531'
# result_file = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/LAVIS-kttian-VIT/lavis/output/BLIP2/Finetune/ALBEF_PTOPT/Findings_wval/20230530012/ckpt29/20230531232/result/test_epochbest.json'
# result_file = os.path.join(LAVIS_ROOT, eval_output_dir, f'lavis/output/BLIP2/Finetune/ALBEF_PTOPT/Findings_wval/20230530012/ckpt29_200tok/20230531165/result/test_epochbest.json')
metrics_cmd = f"python run_metrics_arg.py --cfg-path {eval_config_path} --eval_mode '{eval_mode}' --job_id '{job_id}' --result_file '{result_file}' --date '{date}'"

# %%


# %%
print(metrics_cmd)

# %%



