#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#


import argparse

parser = argparse.ArgumentParser(description="Training a critic model for code generation")
parser.add_argument('--model', default='codet5-base', type=str, help='type of transformers model as model backbone')
parser.add_argument('--model_path', default=None, type=str, help='path to model backbone pretrained weights') 
parser.add_argument('--save_dir', default=None, type=str, help='path to save trained critic model checkpoints') 

# Dataloading
parser.add_argument('--train-path', default='data/APPS/train/', type=str, help='path to training data')
parser.add_argument('--sample-mode', default='uniform_sol', help='sampling output programs following a uniform distribution by program population')

# Model
parser.add_argument('--tuning_mode', default='critic', type=str, help='tuning mode for training LMs')
parser.add_argument('--relative_returns', default=False, action='store_true', help='use relative returns against a baseline during RL')
parser.add_argument('--clone_rl_head', default=False, action='store_true', help='Optional: clone a seperate linear layer for RL samples and initialize it from finetuned LM head')


# Training
parser.add_argument('--epochs', default=10, type=int, help='total number of training epochs')
parser.add_argument('--lr', default=5e-5, type=float, help='training learning rate')
parser.add_argument('--batch-size-per-replica', default=4, type=int, help='batch size per GPU')
parser.add_argument('--grad-acc-steps', default=8, type=int, help='number of training steps before each gradient update')
parser.add_argument('--deepspeed', default = None, type=str, help='path to deepspeed configuration file; set None if not using deepspeed')
parser.add_argument('--fp16', default=True, action='store_true', help='set 16-bit training to reduce memory usage')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--db', default=False, action='store_true', help='set to turn on debug mode i.e. using dummy small data split and only 1 data worker')

# Logging
parser.add_argument('--log-freq', default=1, type=int, help='save training log after this number of training steps')
parser.add_argument('--save-freq', default=200, type=int, help='save model checkpoints after this number of training steps')
parser.add_argument('--save_total_limit', default=2, type=int, help='total of number checkpoints to keep; only keep the latest ones') 

args = parser.parse_args()

if args.save_dir is None: 
     args.save_dir = '{}_{}_bs{}x{}_lr{}'.format(
         args.model, args.tuning_mode,
         args.batch_size_per_replica, args.grad_acc_steps, args.lr
     )

if args.db:
    args.save_dir = 'exps/test/{}'.format(args.save_dir)
else:
    args.save_dir = 'exps/{}'.format(args.save_dir)
