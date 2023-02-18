'''
instructions are outdated TODO

First, 
cd /n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/export/home/.cache/lavis/coco/annotations/
copy each file from coco_karpathy_train.json to coco_karpathy_train_orig.json
create subset files by running this python script

TO RUN UNDER SUBSET CONDITIONS:
cp coco_karpathy_train_subset.json coco_karpathy_train.json
cp coco_karpathy_test_subset.json coco_karpathy_test.json
cp coco_karpathy_val_subset.json coco_karpathy_val_subset.json

TO RUN UNDER ORIGINAL CONDITIONS:
cp coco_karpathy_train_orig.json coco_karpathy_train.json
cp coco_karpathy_test_orig.json coco_karpathy_test.json
cp coco_karpathy_val_orig.json coco_karpathy_val_subset.json
'''

import argparse 
import json 
import os 
import shutil 

cache_root = "/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/export/home/.cache/lavis/"

parser = argparse.ArgumentParser()

parser.add_argument('--init', action='store_true', 
                    help='set to true only if running for the first time & need to copy original data files -- never run again!')
parser.add_argument('--create', action='store_true', 
                    help='if true, will create a new data subset')
parser.add_argument('--save_subset', action='store_true',
                    help='if true, will save the created subset to a _subset file') 
parser.add_argument('--load_small', action='store_true',
                    help='if true, will ready the subset file for use')
parser.add_argument('--load_orig', action='store_true',
                    help='if true, will ready the original full file for use')


def get_filename(split, mode=None, gt=False):
    '''
    split: 'train', 'val', or 'test'
    mode: 'orig', 'subset', or 'none'
    gt: True or False 
    '''
    if mode is None:
        if not gt:
            return os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}.json')
        else:
            return os.path.join(cache_root, 'coco_gt', f'coco_karpathy_{split}_gt.json')
    if not gt:
        return os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_{mode}.json')
    else:
        return os.path.join(cache_root, 'coco_gt', f'coco_karpathy_{split}_gt_{mode}.json')


# coco_karpathy_test.json  coco_karpathy_train.json  coco_karpathy_val.json
# /n/data1/hms/dbmi/rajpurkar/lab/home/kt220/SPR23/export/home/.cache/lavis/
def create_subset(save_subset):
    print("cache root", cache_root)
    # for each coco data split 
    splits = ['train', 'val', 'test']
    subset_sizes = {'train': 64, 'val': 8, 'test': 8} # original sizes: 566747, 5000, 5000
    for split in splits:
        size = subset_sizes[split]
        print("-" * 40)
        print(split)

        # load original data 
        file_name = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_orig.json')
        obj = open(file_name)
        data = json.load(obj)

        print(len(data))
        print(data[1])

        # create data subset 
        data_subset = data[:size]
        new_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_subset.json')
        # save data subset 
        if save_subset:
            with open(new_file, "w") as outfile:
                json.dump(data_subset, outfile)
        
        if split == 'val' or split == 'test':
            # create subset for gt
            gt_file = os.path.join(cache_root, 'coco_gt', f'coco_karpathy_{split}_gt_orig.json')
            new_gt_file = os.path.join(cache_root, 'coco_gt', f'coco_karpathy_{split}_gt_subset.json')
            gt_obj = open(gt_file)
            gt_data = json.load(gt_obj) # keys are 'annotations', 'images'
            print(gt_data["annotations"][0])
            print(gt_data["images"][0])


            new_annotations = [] 
            new_images = []
            # new_images = gt_data['images'][:size] # seems to be in order, but we will double check instead 

            # for each item in the val set, search its id in the gt file to put into the gt subset
            for k in range(size):
                temp = data_subset[k]['image'].split('.')
                id_str = temp[-2][-6:]
                print(temp, "-->", id_str)
                
                # search through annotations 
                found_annot = False 
                for item in gt_data['annotations']:
                    if str(item['image_id']) == id_str:
                        found_annot = True 
                        new_annotations.append(item)
                if found_annot:
                    print(f"found {id_str} in annotations!")
                else:
                    print(f"FAILED to find {id_str} in annotations")

                found_img = False 
                for item in gt_data['images']:
                    if str(item['id']) == id_str:
                        found_img = True
                        new_images.append(item)
                if found_img:
                    print(f"found {id_str} in images!")
                else:
                    print(f"FAILED to find {id_str} in images")
        
            new_gt_data = {'annotations': new_annotations, 'images': new_images}
            print(new_gt_data)
            # save data subset 
            if save_subset:
                with open(new_gt_file, "w") as outfile:
                    json.dump(new_gt_data, outfile)

def download_coco_gt():
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

if __name__ == "__main__":
    args = parser.parse_args()

    if args.init:
        print("first time run - copy file to _orig")
        # copy coco files 
        for split in ['train', 'val', 'test']:
            coco_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}.json')
            orig_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_orig.json')
            if split == 'train':
                coco_obj = open(coco_file)
                coco_data = json.load(coco_obj)
                assert(len(coco_data)==566747)
            shutil.copy(coco_file, orig_file) # src, dest
        
        # copy gt files ... warning: they may not be downloaded yet
        # download_coco_gt() 
        for split in ['val', 'test']:
            coco_file = os.path.join(cache_root, 'coco_gt', f'coco_karpathy_{split}_gt.json')
            orig_file = os.path.join(cache_root, 'coco_gt', f'coco_karpathy_{split}_gt_orig.json')
            if split == 'val':
                coco_obj = open(coco_file)
                coco_data = json.load(coco_obj)
                assert(len(coco_data['images'])==5000)
            shutil.copy(coco_file, orig_file) # src, dest

    if args.create:
        print("create subset")
        create_subset(args.save_subset)
    
    if args.load_orig:
        print("save orig coco dataset")
        for split in ['train', 'val', 'test']:
            used_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}.json')
            orig_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_orig.json')
            shutil.copy(orig_file, used_file)

        for split in ['val', 'test']:
            used_file = os.path.join(cache_root, 'coco_gt', f'coco_karpathy_{split}_gt.json')
            orig_file = os.path.join(cache_root, 'coco_gt', f'coco_karpathy_{split}_gt_orig.json')
            shutil.copy(orig_file, used_file)
    
    if args.load_small:
        print("save small coco subset")
        for split in ['test', 'val', 'train']:
            used_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}.json')
            new_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_subset.json')
            shutil.copy(new_file, used_file)
        
        for split in ['val', 'test']:
            used_file = os.path.join(cache_root, 'coco_gt', f'coco_karpathy_{split}_gt.json')
            new_file = os.path.join(cache_root, 'coco_gt', f'coco_karpathy_{split}_gt_subset.json')
            shutil.copy(new_file, used_file)
    
    print("finished make_coco_subset.")
