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

import os 
import json 

cache_root = "/export/home/.cache/lavis/"
print(cache_root)

splits = ['test', 'val', 'train']
subset_sizes = [5,5,50] # original sizes: 5000, 5000, 566747
for i in range(3):
    split = splits[i]
    size = subset_sizes[i]

    file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_orig.json')
    obj = open(file)
    data = json.load(obj)


    print(len(data))
    print(data[1])

    data_subset = data[:size]
    new_file = os.path.join(cache_root, 'coco', 'annotations', f'coco_karpathy_{split}_subset.json')
    with open(new_file, "w") as outfile:
        json.dump(data_subset, outfile)

# coco_karpathy_test.json  coco_karpathy_train.json  coco_karpathy_val.json
# /export/home/.cache/lavis/