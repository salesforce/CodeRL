#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

# Run code with deepspeed 
USE_TF=NO deepspeed --master_port 62000 \
    train.py \
    --batch-size-per-replica 8 --grad-acc-steps 1 \
    --epochs 10 --lr 2e-5 \
    --save-freq 1000 --log-freq 10 --save_total_limit 5 \
    --tuning_mode critic --model codet5-base \
    --fp16 --deepspeed configs/deepspeed_configs.json 
    