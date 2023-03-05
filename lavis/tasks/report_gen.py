"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask



@registry.register_task("report_gen")
class ReportGenTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=False):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)
        print("setup_task: report_metric", report_metric)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )
    
    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )
        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": int(img_id)})
        return results

    def valid_step_two_token(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        model.set_prompt('<findings> ')
        findings_strs = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )
        model.set_prompt('<impression> ')
        impression_strs = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        for finding_str, impression_str, img_id in zip(findings_strs, impression_strs, img_ids):
            results.append({"findings": finding_str, "impression": impression_str, "image_id": int(img_id)})
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        res = {} 
        res["dummy_ten"] = 10
        res["agg_metrics"] = 1 #agg_metrics is needed

        return res

    def _report_metrics2(self, eval_result_file, split_name):
        # load the generated reports 
        f = open(eval_result_file)
        eval_result = json.load(f)

        # load the ground truth reports 
        # TODO (hard-coded paths for now, but will retrieve from dataset config later)
        def get_filename(split):
            outdir = os.path.join(registry.get_path("cache_root"), 'mimic_cxr', 'annotations')
            name = f'mimic_{split}_impression_subset.json'
            outfile = os.path.join(outdir, name)
            return outfile 
        
        gt_filename = get_filename(split_name)
        gt_f = open(gt_filename)
        gt_json = json.load(gt_f)

        my_metric = kat_eval_json(eval_result, gt_json)

        res = {} 
        res["kat_metric"] = my_metric 
        res["five"] = 5
        res["agg_metrics"] = my_metric #agg_metrics is needed

        return res

def kat_eval_json(gen_json, gt_json):
    # assert(len(gen_json) == len(gt_json))
    gen_dict = {}
    gt_dict = {}
    for item in gen_json:
        # the keys were ints 
        gen_dict[str(item["image_id"])] = item["caption"]
    
    for item in gt_json:
        # the keys were all strings 
        gt_dict[str(item["image_id"])] = item["caption"][0] # since val is a list 
    
    # assert(gt_dict.keys() == gen_dict.keys())

    my_metric = 0 
    for key in gt_dict.keys():
        gt_caption = gt_dict[key]
        gen_caption = gen_dict[key]

        gt_words = gt_caption.split(" ")
        gen_words = gen_caption.split(" ")


        gt_word_set = set(gt_words)
        count = 0
        for gen_wd in gen_words:
            if gen_wd in gt_word_set:
                count += 1 
        acc = count / len(gen_words)
        my_metric += acc 
    
    return my_metric / (len(gt_dict))

# def kat_eval(gts, res):
#     assert(gts.keys() == res.keys())

from pycocoevalcap.bleu.bleu import Bleu 
from pycocoevalcap.meteor.meteor import Meteor


# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url

# https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py 
def coco_caption_eval(coco_gt_root, results_file, split):
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

    # create coco object and coco_result object
    print("annotation file:", annotation_file)
    print("results file:", results_file)
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
