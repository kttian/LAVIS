"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.logger import MetricLogger
import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized
from lavis.datasets.data_utils import prepare_sample


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass
    # def evaluation(self, model, data_loader, cuda_enabled=True):
    #     metric_logger = MetricLogger(delimiter="  ")
    #     header = "Evaluation"
    #     # TODO make it configurable
    #     print_freq = 10

    #     results = []

    #     for samples in metric_logger.log_every(data_loader, print_freq, header):
    #         samples = prepare_sample(samples, cuda_enabled=cuda_enabled) # batch of image ids

    #         eval_output = self.valid_step(model=model, samples=samples)
    #         results.extend(eval_output)

    #     if is_dist_avail_and_initialized():
    #         dist.barrier()

    #     return results

    # def valid_step(self, model, samples):
    #     # use_amp = scaler is not None
    #     with torch.cuda.amp.autocast(enabled=True):
    #         loss = self.train_step(model=model, samples=samples)
    #         return loss.item()
    
    # def after_evaluation(self, val_result, split_name, epoch):
    #     return val_result 