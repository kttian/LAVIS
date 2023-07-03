"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.caption_datasets import (
    CaptionDataset,
    CaptionEvalDataset,
)
from lavis.datasets.datasets.report_gen_datasets import (
    ReportGenDataset,
    ReportGenEvalDataset,
    ReportGenBothDataset,
    ReportGenBothEvalDataset,
    ReportGenInstructDataset,
    ReportGenInstructEvalDataset,
)
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry


### instruct datasets ### 

@registry.register_builder("mimic_cxr_instruct_target_wval_subset_224")
class MimicCxrInstructTargetWvalSubset224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_target_wval_subset_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_target_wval_full_224")
class MimicCxrInstructTargetWvalFull224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_target_wval_full_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_target_noval_subset_224")
class MimicCxrInstructTargetNovalSubset224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_target_noval_subset_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_target_noval_full_224")
class MimicCxrInstructTargetNovalFull224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_target_noval_full_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_findings_wval_subset_224")
class MimicCxrInstructFindingsWvalSubset224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_findings_wval_subset_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_findings_wval_full_224")
class MimicCxrInstructFindingsWvalFull224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_findings_wval_full_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_findings_noval_subset_224")
class MimicCxrInstructFindingsNovalSubset224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_findings_noval_subset_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_findings_noval_full_224")
class MimicCxrInstructFindingsNovalFull224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_findings_noval_full_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_impression_wval_subset_224")
class MimicCxrInstructImpressionWvalSubset224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_impression_wval_subset_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_impression_wval_full_224")
class MimicCxrInstructImpressionWvalFull224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_impression_wval_full_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_impression_noval_subset_224")
class MimicCxrInstructImpressionNovalSubset224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_impression_noval_subset_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_impression_noval_full_224")
class MimicCxrInstructImpressionNovalFull224Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_impression_noval_full_224.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_target_wval_subset_364")
class MimicCxrInstructTargetWvalSubset364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_target_wval_subset_364.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_target_wval_full_364")
class MimicCxrInstructTargetWvalFull364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_target_wval_full_364.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_target_noval_subset_364")
class MimicCxrInstructTargetNovalSubset364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_target_noval_subset_364.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_target_noval_full_364")
class MimicCxrInstructTargetNovalFull364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_target_noval_full_364.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_findings_wval_subset_364")
class MimicCxrInstructFindingsWvalSubset364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_findings_wval_subset_364.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_findings_wval_full_364")
class MimicCxrInstructFindingsWvalFull364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_findings_wval_full_364.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_findings_noval_subset_364")
class MimicCxrInstructFindingsNovalSubset364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_findings_noval_subset_364.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_findings_noval_full_364")
class MimicCxrInstructFindingsNovalFull364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_findings_noval_full_364.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_impression_wval_subset_364")
class MimicCxrInstructImpressionWvalSubset364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_impression_wval_subset_364.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_impression_wval_full_364")
class MimicCxrInstructImpressionWvalFull364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_impression_wval_full_364.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_impression_noval_subset_364")
class MimicCxrInstructImpressionNovalSubset364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_impression_noval_subset_364.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_impression_noval_full_364")
class MimicCxrInstructImpressionNovalFull364Builder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr_0529/mimic_cxr_instruct_impression_noval_full_364.yaml",
    }


### sepna wval datasets ### 
@registry.register_builder("mimic_cxr_impression_sepna_wval_224")
class MimicCxrImpressionSepnaWval224Builder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/sepna_wval_impression_full_224.yaml",
    }

@registry.register_builder("mimic_cxr_findings_sepna_wval_224")
class MimicCxrFindingsSepnaWval224Builder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/sepna_wval_findings_full_224.yaml",
    }


### sepna noval datasets ### 
# TODO

#### old datasets ##### 
### not sepna, wval datasets ### 
@registry.register_builder("mimic_cxr_findings_wval")
class MimicCxrFindingsWvalBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/findings_full_wval.yaml",
    }

@registry.register_builder("mimic_cxr_impression_subset_wval")
class MimicCxrImpressionSubsetWvalBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/impression_subset_wval.yaml",
    }

@registry.register_builder("mimic_cxr_findings_subset_wval")
class MimicCxrFindingsSubsetWvalBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/findings_subset_wval.yaml",
    }

### not sepna, not wval ### 
@registry.register_builder("mimic_cxr_impression")
class MimicCxrImpressionBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/impression_full.yaml",
    }

@registry.register_builder("mimic_cxr_findings")
class MimicCxrFindingsBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/findings_full.yaml",
    }

@registry.register_builder("mimic_cxr_impression_subset")
class MimicCxrImpressionSubsetBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/impression_subset.yaml",
    }

@registry.register_builder("mimic_cxr_findings_subset")
class MimicCxrFindingsSubsetBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/findings_subset.yaml",
    }


### old
@registry.register_builder("mimic_cxr")
class MimicCxrBuilder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenBothDataset
    eval_dataset_cls = ReportGenBothEvalDataset
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/defaults.yaml",
    }

@registry.register_builder("mimic_cxr_subset")
class MimicCxrSubsetBuilder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenBothDataset 
    eval_dataset_cls = ReportGenBothEvalDataset
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/defaults_subset.yaml",
    }