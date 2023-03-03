"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.caption_datasets import (
    CaptionDataset,
    CaptionEvalDataset
)
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)


@registry.register_builder("mimic_cxr")
class MimicCxrBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/defaults.yaml",
    }


@registry.register_builder("mimic_cxr_subset")
class MimicCxrSubsetBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset # can use the default caption dataset class for now 
    eval_dataset_cls = CaptionEvalDataset # use default eval dataset for now as well 
    # check lavis/datasets/datasets/coco_caption_datasets.py later 

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/defaults_subset.yaml",
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

@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }
