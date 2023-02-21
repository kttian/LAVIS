'''
refer to /n/data1/hms/dbmi/rajpurkar/lab/home/kt220/albef/ALBEF_Preprocessing_Splits.py on O2 
refer to https://github.com/rajpurkarlab/report-generation-baseline/blob/main/datasets.py#L427 

under export/home/.cache/lavis:
- coco
- mimic_cxr
-- annotations
--- mimic_train_impression.json
--- mimic_val_impression.json
--- mimic_test_impression.json
--- mimic_train_subset.json
--- mimic_val_subset.json
--- mimic_test_subet.json
-- images (symlink to /n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/mimic-cxr-resized-256/2.0.0/files)

I temporarily copy the test file into the val file, since we don't have a val split right now. 
'''

import argparse 
import json 
import os
import pandas as pd 

cache_root = "/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/export/home/.cache/lavis/"

parser = argparse.ArgumentParser()
parser.add_argument('--report_mode', type=str, default='impression', help='whether to generate findings, impression, or both')
parser.add_argument('--size_mode', type=str, default='subset', help='whether to generate mimic subset or full')
parser.add_argument('--image', action='store_true') 
parser.add_argument('--save', action='store_true') 
parser.add_argument('--val', action='store_true') 


data_train = pd.read_csv('mimic_train_reports.csv')
data_test = pd.read_csv('mimic_test_reports.csv')
data = {'train': data_train, 'test': data_test}


def convert_to_path(row):
    ''' 
    assumes input is a row of the dataframe
    ''' 
    dicom_id = row['dicom_id']
    subject_id = str(int(row['subject_id']))
    study_id = str(int(row['study_id']))
    img_file = f"p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.png"

    image_id = "".join([str(int(x, 16)) for x in dicom_id.split('-')])
    return img_file, dicom_id, f"p{subject_id}", f"s{study_id}", image_id 


def get_report(row, report_mode):
    if report_mode == 'findings' or report_mode == 'impression':
        return row[report_mode]
    else: # both
        # TODO: what is the proper way to generate both?
        return f"FINDINGS: {row['findings']} \n IMPRESSION: {row['impression']}"


def process_data(data_df):
    data_df.dropna()


def get_filename(split, report_mode, size_mode):
    outdir = os.path.join(cache_root, 'mimic_cxr', 'annotations')
    name = f'mimic_{split}_{args.report_mode}_{args.size_mode}.json'
    outfile = os.path.join(outdir, name)
    return outdir, outfile 


def main(args):
    subset_sizes = {'train': 64, 'val': 8, 'test': 8} # original sizes: ????

    # create annotations 
    for split in ['train', 'test']:
        print('-'*40)
        print(split)
        size = subset_sizes[split]
        
        # create file path of where to save annotations json 
        outdir, outfile = get_filename(split, args.report_mode, args.size_mode)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        # prepare data df for subset 
        # TODO: change data processing for full data later
        data_split = data[split].dropna()
        data_subset = data_split[:size]

        # loop through df and create json data 
        json_data = []
        for i, row in data_subset.iterrows():
            img_file, dicom_id, subject_id, study_id, image_id = convert_to_path(row)
            report = get_report(row, args.report_mode) 
            if split == 'train':
                report_item = report 
            else:
                report_item = [report]
            json_item = {'image': img_file, 'caption': report_item, 'dicom_id': dicom_id, 
                         'subject_id': subject_id, 'study_id': study_id, 'image_id': image_id}
            json_data.append(json_item)
        
        if args.save:
            with open(outfile, "w") as f:
                json.dump(json_data, f)
            print(f"saved to {outfile}")
        else:
            print(json_data)
    
    # handle images by symlinking the mimic parent to mimic_cxr/images
    if args.image:
        imgdir = os.path.join(cache_root, 'mimic_cxr', 'images')
        os.system(f'ln -s /n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/mimic-cxr-resized-256/2.0.0/files/ {imgdir}')
    
    if args.val:
        # copy from train to val
        _, train_path = get_filename('train', args.report_mode, args.size_mode)
        _, val_path = get_filename('val', args.report_mode, args.size_mode)
        # read train json
        f = open(train_path, 'r')
        train_json = json.load(f)
        for item in train_json:
            item['caption'] = [item['caption']]
        # write to val json
        with open(val_path, 'w') as fv:
            json.dump(train_json, fv)
        # copy from test to val
        # _, test_path = get_filename('test', args.report_mode, args.size_mode)
        # _, val_path = get_filename('val', args.report_mode, args.size_mode)
        # os.system(f'cp {test_path} {val_path}')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args) 