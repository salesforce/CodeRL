#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# 
import json
import os
import pprint
import torch
import pdb 
import glob 
from tqdm import tqdm
import pickle as pkl 
import numpy as np 
from collections import Counter 
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import datasets.utils as dsutils

def generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, 
                    starter_path=None):
    
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data 
        _input += data
    
    if os.path.exists(test_case_path):
        with open(test_case_path, "r") as f:
            data = json.load(f)
        if not data.get("fn_name"):
            _input += "\nUse Standard Input format"
        else:
            _input += "\nUse Call-Based format"
    elif starter_path is not None and os.path.exists(starter_path):
        _input += "\nUse Call-Based format"
    else:
        _input += "\nUse Standard Input format"
        
    _input += "\nANSWER:\n"
    
    return _input

def generate_critic_inputs(args, test_case_path, prompt_path, solutions_path, tokenizer, 
                           starter_path=None, gt_solutions=False):    
    _input = generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path)
    
    q_tokens = tokenizer.encode(_input, verbose=False, max_length=args.source_len)
    in_tokens = [tokenizer.eos_token_id] * args.source_len
    in_tokens[:len(q_tokens)] = q_tokens
    in_tokens = in_tokens[:args.source_len]
    
    solutions = json.load(open(solutions_path, 'r')) 
    
    all_texts = []
    gt_errors = [] 
    all_codes = [] 
    
    for sol_index, solution in enumerate(solutions):        
        if gt_solutions: 
            solution_str = dsutils.reindent_code(solution)
        else:
            solution_str = dsutils.reindent_code(solution['code'])
            
        a_tokens = tokenizer.encode(solution_str)        
        code = [-100] * args.max_len 
        code[:len(a_tokens)] = a_tokens         
        code = code[:args.max_len]
            
        all_texts.append(in_tokens)
        all_codes.append(code)
        
        if gt_solutions: 
            gt_errors.append(dsutils.get_error_type(True))
        else:
            gt_errors.append(dsutils.get_error_type(solution['result']))

    return all_texts, all_codes, gt_errors

def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    original_problems = glob.glob(args.test_path + '/*')
    problems = sorted(original_problems) 

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    print("Saving results to {}".format(args.output_path))

    if args.start > len(problems) or args.start < 0:
        print(f"start index {args.start} > number of problems {len(problems)}")
        return
    start = args.start
    if args.end is None or args.end > len(problems):
        end = len(problems)
    else:
        end = args.end
    problems = problems[start:end]
    
    # Set up model
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base', cache_dir=args.tokenizer_path)
    print("Loading model from {}...".format(args.model_path))
    if args.critic_scores:
        model = T5ForConditionalGeneration.from_pretrained(args.model_path, tuning_mode='critic') 
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model_path) 
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
   
    if args.critic_scores:
        all_preds = [] 
        all_gts = [] 
        
    # main eval loop
    for index, problem in tqdm(enumerate(problems), ncols=0, total=len(problems)):
        
        prob_path = os.path.join(problem)
        print(f"problem path = {prob_path}")
        
        problem_id = int(problem.split('/')[-1])
        
        if args.critic_scores and \
            os.path.exists(os.path.join(args.output_path, f"{problem_id}_gt{args.gt_solutions}.pkl")):
            continue 
        elif os.path.exists(os.path.join(args.output_path, f"{problem_id}.json")):
            continue 
        
        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        if args.critic_scores and not args.gt_solutions: 
            solutions_path = os.path.join(prob_path, "gen_solutions.json")
        else:
            solutions_path = os.path.join(prob_path, "solutions.json")
        if not os.path.exists(starter_path):
            starter_path = None

        if args.critic_scores:
            input_texts, input_codes, gt_error_types = generate_critic_inputs(args, test_case_path, prompt_path, solutions_path,
                                                                  tokenizer, starter_path, args.gt_solutions)
        else:
            input_text = generate_prompt(args, test_case_path, prompt_path, solutions_path, 
                                          tokenizer, starter_path)

        with torch.no_grad():
            if args.critic_scores:
                text_tensor = torch.tensor(input_texts).to(device)
                code_tensor = torch.tensor(input_codes).to(device)
                gt_error_tensor = torch.tensor(gt_error_types).to(device)
                
                curr_inputs = {'input_ids': text_tensor, 'error_types': gt_error_tensor, 'labels': code_tensor}
                _, error_preds, error_hidden_states = model(**curr_inputs, return_error_hidden_states=True)
                
                assert len(gt_error_types) == len(error_preds)
                all_preds.extend(error_preds.cpu().numpy().tolist())
                all_gts.extend(gt_error_types)
                
                saved_critic_scores = {}
                saved_critic_scores[problem_id] = {'code': input_codes, 'prompt': input_texts,
                                          'gt_error_type': gt_error_types, 
                                          'pred_error_type': error_preds.cpu().numpy(),
                                          'error_hidden_states': error_hidden_states.cpu().numpy()}
                scores_loc = os.path.join(args.output_path,  f"{problem_id}_gt{args.gt_solutions}.pkl")
                pkl.dump(saved_critic_scores, open(scores_loc, 'wb'))
                    
            else:
                input_ids = torch.LongTensor(tokenizer.encode(input_text, 
                                                              verbose=False, 
                                                              max_length=args.source_len)).unsqueeze(0).cuda()

                num_loops = int(args.num_seqs / args.num_seqs_per_iter)
                output_programs = [] 
                for i in tqdm(range(num_loops), ncols=0, total=num_loops, leave=False):
                    output_ids = model.generate(
                        input_ids, 
                        do_sample=True, 
                        temperature=args.temperature, 
                        max_length=args.max_len, 
                        num_return_sequences=args.num_seqs_per_iter,
                        top_p=0.95)                    

                    for output_id in output_ids: 
                        output_programs.append(tokenizer.decode(output_id, skip_special_tokens=True))

                saved_codes = {}
                saved_codes[problem_id] = {'code': output_programs, 'prompt': input_text}

                codes_loc = os.path.join(args.output_path, f"{problem_id}.json")
                with open(codes_loc, "w") as f:
                    json.dump(saved_codes, f)

    if args.critic_scores: 
        print("Total number of samples: {}".format(len(all_gts)))
        acc = (np.array(all_preds) == np.array(all_gts)).sum()/len(all_gts)
        print("Error Pred Acc: {}".format(acc))
        print("Prediction distribution: {}".format(Counter(all_preds)))
        print("GT distribution: {}".format(Counter(all_gts)))
                    
if __name__ == "__main__":
    
    from configs.generate_configs import * 
    
    main(args)
