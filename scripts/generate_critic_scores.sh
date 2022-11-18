##
## Copyright (c) 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
## 
critic_path=models/codet5_finetuned_critic/
tokenizer_path=models/codet5_tokenizer/
test_path=data/APPS/train/ #test.json

output_path=data/APPS/train/

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_path ${critic_path} \
    --test_path ${test_path} \
    --output_path ${output_path} \
    --critic_scores --gt_solutions  
