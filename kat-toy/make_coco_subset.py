'''
First, 
cd /export/home/.cache/lavis/coco/annotations/
copy each file from coco_karpathy_train.json to coco_karpathy_train_orig.json
create subset files by running this python script

TO RUN UNDER SUBSET CONDITIONS:
mv coco_karpathy_train_subset.json coco_karpathy_train.json
mv coco_karpathy_test_subset.json coco_karpathy_test.json
mv coco_karpathy_val_subset.json coco_karpathy_val_subset.json

TO RUN UNDER ORIGINAL CONDITIONS:
mv coco_karpathy_train_orig.json coco_karpathy_train.json
mv coco_karpathy_test_orig.json coco_karpathy_test.json
mv coco_karpathy_val_orig.json coco_karpathy_val_subset.json
'''

import argparse 
import json 
import os 
import shutil 

cache_root = "/export/home/.cache/lavis/"

parser = argparse.ArgumentParser()
parser.add_argument('--create', action='store_true')
parser.add_argument('--save_small', action='store_true')
parser.add_argument('--save_orig', action='store_true')

# coco_karpathy_test.json  coco_karpathy_train.json  coco_karpathy_val.json
# /export/home/.cache/lavis/
def create_subset():
    print("cache root", cache_root)
    # for each coco data split 
    splits = ['test', 'val', 'train']
    subset_sizes = [8,8,64] # original sizes: 5000, 5000, 566747
    for i in range(3):
        split = splits[i]
        size = subset_sizes[i]

        # load original data 
        file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_orig.json')
        obj = open(file)
        data = json.load(obj)

        print(len(data))
        print(data[1])

        # create data subset 
        data_subset = data[:size]
        new_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_subset.json')
        # save data subset 
        with open(new_file, "w") as outfile:
            json.dump(data_subset, outfile)
    

if __name__ == "__main__":
    args = parser.parse_args()

    if args.create:
        create_subset()
    
    if args.save_orig:
        print("save orig coco dataset")
        for split in ['test', 'val', 'train']:
            used_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}.json')
            orig_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_orig.json')
            shutil.copy(orig_file, used_file)
    
    if args.save_small:
        print("save small coco subset")
        for split in ['test', 'val', 'train']:
            used_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}.json')
            new_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_subset.json')
            shutil.copy(new_file, used_file)
    
    print("finished make_coco_subset.")
