#
# '''
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# '''#
import json
import os
import pprint
import torch
import pdb 
import glob 
from tqdm import tqdm
from transformers import RobertaTokenizer, T5ForConditionalGeneration

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
    model = T5ForConditionalGeneration.from_pretrained(args.model_path) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
   
    # main eval loop
    for index, problem in tqdm(enumerate(problems), ncols=0, total=len(problems)):
        
        prob_path = os.path.join(problem)
        print(f"problem path = {prob_path}")
        
        problem_id = int(problem.split('/')[-1])
        if os.path.exists(os.path.join(args.output_path, f"{problem_id}.json")):
            continue 

        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")
        if not os.path.exists(starter_path):
            starter_path = None

        input_text = generate_prompt(args, test_case_path, prompt_path, solutions_path, 
                                          tokenizer, starter_path)

        with torch.no_grad():
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
    
if __name__ == "__main__":
    
    from configs.generate_configs import * 
    
    main(args)
