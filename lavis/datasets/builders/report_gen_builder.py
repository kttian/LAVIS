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


@registry.register_builder("mimic_cxr_instruct_findings_wval")
class MimicCxrInstructFindingsSubsetBuilder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/instruct_findings_wval.yaml",
    }

@registry.register_builder("mimic_cxr_instruct_findings_subset_wval")
class MimicCxrInstructFindingsSubsetBuilder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenInstructDataset
    eval_dataset_cls = ReportGenInstructEvalDataset
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/instruct_findings_subset_wval.yaml",
    }

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

# new impression dataset with val set
@registry.register_builder("impression")
class MimicCxrImpressionWvalBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/sepna_wval_impression_full_224.yaml",
    }


# new impression dataset with val set, separate na filters, size 224
@registry.register_builder("mimic_cxr_impression_sepna_wval_224")
class MimicCxrImpressionSepnaWval224Builder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/sepna_wval_impression_full_224.yaml",
    }

# new findings dataset with val set, separate na filters, size 224
@registry.register_builder("mimic_cxr_findings_sepna_wval_224")
class MimicCxrFindingsSepnaWval224Builder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/sepna_wval_findings_full_224.yaml",
    }

# new findings dataset with val set 
@registry.register_builder("mimic_cxr_findings_wval")
class MimicCxrFindingsWvalBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/findings_full_wval.yaml",
    }

# new impression dataset with val set
@registry.register_builder("mimic_cxr_impression_subset_wval")
class MimicCxrImpressionSubsetWvalBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/impression_subset_wval.yaml",
    }

# new findings dataset with val set 
@registry.register_builder("mimic_cxr_findings_subset_wval")
class MimicCxrFindingsSubsetWvalBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/findings_subset_wval.yaml",
    }

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