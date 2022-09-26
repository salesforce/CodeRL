#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import torch
import glob
import logging
import random
import fnmatch
import numpy as np
import gc
import os
from tqdm import tqdm 
from collections import Counter
import pickle as pkl 
import json, pdb 

from multiprocessing import Manager
import transformers

import datasets.utils as dsutils

class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problem_dirs, model, max_tokens, sample_mode, 
                 tuning_mode, max_src_tokens):
        self.dataroot = dataroot
        self.problem_dirs = problem_dirs 

        self.model = model
        self.sample_mode = sample_mode
        self.tuning_mode = tuning_mode
        
        self.max_tokens = max_tokens
        self.max_src_tokens = max_src_tokens

        self.samples = []           
        self.all_error_types, self.all_error_subtypes = [], [] 
        self.initialize()

        if self.model in ['codet5-base', 'codet5-large']:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
       
    def load_gen_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        info = []
        
        for idx, sol in enumerate(sols):
            sol_str = sol['code']  
            sample = (question_str, starter_code, sol_str, answer_type)
            samples.append(sample) 
            
            result = sol['result']
            error_type = sol['error_type']
            
            info.append((result, error_type))
            
        return samples, info 
     
    def get_gt_info(self):
        return (1, None)
    
    def load_gt_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        
        for sol_str in sols:
            sol_str = dsutils.reindent_code(sol_str)
            sample = (question_str, starter_code, sol_str, answer_type)
            samples.append(sample)
        
        return samples 
    
    def update_error_stat(self, info):
        for i in info:
            error_type = dsutils.get_error_type(i[0])
            error_subtype = i[1]
            self.all_error_types.append(error_type)
            self.all_error_subtypes.append(error_subtype) 
    
    def initialize(self):
        all_samples = []
        skipped_problems = []
        samples_info = [] 
        gen_samples = [] 
        
        all_samples_dict = {} 

        print(f"Loading {len(self.problem_dirs)} problems from {self.dataroot}.")
        for problem_name in tqdm(self.problem_dirs):           
            if self.tuning_mode in ['critic']:                
                gen_sols_fname = [os.path.join(self.dataroot, problem_name, "gen_solutions.json")]       

            question_fname = os.path.join(self.dataroot, problem_name, "question.txt")
            sols_fname = os.path.join(self.dataroot, problem_name, "solutions.json")            
            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue
                
            # Read the question description
            with open(question_fname, 'r') as f:
                question_str = f.read()
            
            starter_code = os.path.join(self.dataroot, problem_name, "starter_code.py")    
            if (os.path.isfile(starter_code)):
                answer_type = "\nUse Call-Based format\n"
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                answer_type = "\nUse Standard Input format\n"
                starter_code = ""

            sols_str_list = json.load(open(sols_fname, 'r'))
            gt_samples = self.load_gt_samples(sols_str_list, answer_type, starter_code, question_str)
            all_samples += gt_samples 
                
            # Read all the solutions
            if self.tuning_mode in ['critic']: 
                for fname in gen_sols_fname:
                    if os.path.exists(fname):
                        gen_sols = json.load(open(fname, 'r'))
                        samples, info = self.load_gen_samples(gen_sols, answer_type, starter_code, question_str) 
                        self.update_error_stat(info)
                        gen_samples += samples
                        samples_info += info
                
                # also include ground-truth samples to train critic model; assume success test outcomes 
                gen_samples += gt_samples
                info = [self.get_gt_info() for s in gt_samples]
                samples_info += info
                
                                
        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")
        
        if self.tuning_mode in ['critic']:
            if len(gen_samples) != len(samples_info): pdb.set_trace()
            print(f"Loaded {len(gen_samples)} generated samples from {self.dataroot}.")
            print("Error type distribution: {}".format(Counter(self.all_error_types)))
            print("Error subtype distribution: {}".format(Counter(self.all_error_subtypes)))
        else:
            print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
            
        self.samples = all_samples
        self.samples_info = samples_info 
        self.gen_samples = gen_samples 

    def __len__(self):
        if self.tuning_mode in ['critic']:
            return len(self.gen_samples)
        return len(self.samples)


    def pack_samples(self, idx, sample_type=None):
        """
        Repeatedly pick question, answer pairs from self.dataroot until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the  
        self.question_prefix. These will be added later and the total input will be 
        truncated if necessary.

        Always include the sample at idx at the beginning.
        """
        curr_num_tokens = 0
        curr_samples = [] 
        
        if sample_type == 'gen':
            sample_pool = self.gen_samples
        else:
            sample_pool = self.samples
        
        if self.sample_mode == 'uniform_sol':
            curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[idx]
            if self.tuning_mode in ['critic'] and sample_type=='gen': 
                curr_result, curr_error_subtype = self.samples_info[idx] 
                
        elif self.sample_mode == 'uniform_prob':
            raise NotImplementedError()

        while curr_num_tokens < self.max_tokens:

            curr_q = curr_q[:150000]
            curr_s = curr_s[:150000]
            curr_a = curr_a[:150000]

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))            
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))
            
            if self.tuning_mode in ['critic'] and sample_type=='gen': 
                curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix, 
                                     curr_result, curr_error_subtype))
                break 
     
            else:
                curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix))
                
                # only pack 1 sample each sequence for codeT5 
                if self.model in ['codet5-base', 'codet5-large']:
                    break 

            if self.sample_mode == 'uniform_sol':
                new_idx = random.randint(0, len(sample_pool)-1)
                curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[new_idx] 
            elif self.sample_mode == 'uniform_prob':
                raise NotImplementedError()

        return curr_samples

    def __getitem__(self, idx):
        
        if self.tuning_mode in ['critic']:
            raw_samples = self.pack_samples(idx, 'gen')
            inputs = self.sample_task(raw_samples, 'gen')
         
        else:
            raw_samples = self.pack_samples(idx)
            inputs = self.sample_task(raw_samples)

        gc.collect()
        return inputs
    
    def sample_task(self, samples, sample_type=None):

        input_ids = []
        label_ids = []
                    
        if self.tuning_mode in ['critic'] and sample_type == 'gen': 
            error_types = [] 
                    
        for sample in samples:
            if self.tuning_mode in ['critic'] and sample_type == 'gen': 
                q_str, s_str, a_str, answer_type, result, error_subtype = sample
            else:
                q_str, s_str, a_str, answer_type = sample
            
            q_str =  "\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"

            question_token_ids = self.tokenizer.encode(q_str, verbose=False)
            input_ids.extend(question_token_ids)
             
            answer_token_ids = self.tokenizer.encode(a_str, verbose=False)
            if self.model not in ['codet5-base', 'codet5-large']:
                label_ids.extend([-100] * len(question_token_ids))
                answer_token_ids.append(self.tokenizer.eos_token_id)
                input_ids.extend(answer_token_ids)
            label_ids.extend(answer_token_ids)

            if self.tuning_mode in ['critic'] and sample_type == 'gen':
                error_types.append(dsutils.get_error_type(result))
                
        # Sanity checks and padding 
        input_ids_max_len = self.max_src_tokens if self.model in ['codet5-base', 'codet5-large'] else self.max_tokens 
        if len(input_ids) < input_ids_max_len: 
            new_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len
            new_input_ids[:len(input_ids)] = input_ids
            input_ids = new_input_ids 
            
            if self.model not in ['codet5-base', 'codet5-large']:
                new_label_ids = [-100] * input_ids_max_len 
                new_label_ids[:len(label_ids)] = label_ids
                label_ids = new_label_ids
                
        if self.model in ['codet5-base', 'codet5-large'] and len(label_ids) < self.max_tokens:
            new_label_ids = [-100] * self.max_tokens 
            new_label_ids[:len(label_ids)] = label_ids
            label_ids = new_label_ids
        
        if self.model not in ['codet5-base', 'codet5-large'] and len(input_ids) != len(label_ids): pdb.set_trace()
            
        if self.tuning_mode in ['critic'] and sample_type == 'gen': 
            assert len(error_types) == 1 
            
        # Cut off the excess
        input_ids = input_ids[:input_ids_max_len]
        label_ids = label_ids[:self.max_tokens]
        
        out_sample = {
            "input_ids" : torch.LongTensor(input_ids),
            "labels" :  torch.LongTensor(label_ids)
        }
        
        if self.tuning_mode in ['critic'] and sample_type == 'gen': 
            out_sample['error_types'] = torch.LongTensor(error_types)            
            
        return out_sample 

