from model_apis import *
from task_verifiers import *

import os.path as osp
import json
import PIL.Image
import cv2
import os
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset


GET_ACC = {
    'paperfolding': single_number_accuracy,
    'cube': multi_choice_accuracy,
    'multihop': multihop_accuracy,
    'mmsi': multi_choice_accuracy,
    'ballgame': boxed_accuracy,
    'maze': maze_accuracy,
    'sokoban': sokoban_accuracy,
}

GET_CLINET = None   # TODO: initialze get_client from model_apis.py
GET_RESPONSE = None   # TODO: initialze get_response from model_apis.py
AK = None  # TODO: add your API key here


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='paperfolding')
    parser.add_argument("--model", type=str, default='gemini3pro')
    parser.add_argument("--save_dir", type=str, default='visworld_eval_output')
    parser.add_argument('--save_img', default=False, action='store_true')
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.task, args.model)
    os.makedirs(save_dir, exist_ok=True)

    client = GET_CLINET[args.model](AK) 

    dataset = load_dataset("VisWorld-Eval", split='ballgame')

    total_acc = 0.0
    total = 0
    for line_idx, line in tqdm(enumerate(dataset)):

        output_dir = osp.join(save_dir, str(line_idx))
        response_path = osp.join(output_dir, 'response.json')
        if osp.exists(response_path):
            acc = json.load(open(response_path))['score']

            total_acc += acc
            total += 1
            continue

        prompt, answer, images = line['prompt'], line['answer'], line['image']
        line['extra_info'] = json.loads(line['extra_info'])

        question = line['prompt']
        if line['category'] == 'paperfolding':
            question += "\nPlease answer directly with a single number and nothing else."

        if line['category'] in ['cube', 'mmsi']:
            question += "\nPlease answer directly with only the letter of the correct option and nothing else."

        if line['category'] in ['multihop']:
            question += "\nPlease answer directly with only the letter of the correct option (if applicable), or a single short phrase."

        if line['category'] in ['maze']:
            question += " where x and y are normalized pixel coordinates in the range [0, 1000]. x denotes the horizontal position from left to right, and y denotes the vertical position from top to bottom."

        if line['category'] in ['sokoban']:
            question = question.replace('Present your final movement sequence inside \\bbox{}.\n',
                                        "Present your final movement sequence inside \\bbox{}, where each step must be one of the following directions: up, down, left, right. Separate the steps with commas.\n")

        message = get_message([
            *images,
            question
        ], images)

        response = get_response_until(GET_RESPONSE[args.model], client, message)   # TODO: initialze get_response from model_apis.py
        if args.task == 'maze':
            acc, drawed_gray_array = GET_ACC[args.task](line, response)
        else:
            acc = GET_ACC[args.task](line, response)

        total_acc += acc
        total += 1

        # save
        os.makedirs(output_dir, exist_ok=True)
        if args.save_img and args.task == 'maze':
            cv2.imwrite(osp.join(output_dir, 'visualize.png'), drawed_gray_array)

        with open(response_path, 'w') as f:
            res = {
                'prompt': question,
                'answer': str(answer),
                'response': response,
                'score': acc,
            }
            if 'question_type' in line['extra_info']:
                res['question_type'] = line['extra_info']['question_type']
            json.dump(res, f, indent=4)

        if args.save_img:
            for img_idx, img in enumerate(images):
                image_path = osp.join(output_dir, f'input{img_idx}.png')
                img.save(image_path)

    print(f'Accuracy: {total_acc / total:.4f}')
