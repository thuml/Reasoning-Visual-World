from maze import wall_judge
from sokoban import sokoban_compute_reward

import time
import json
import random
from openai import OpenAI


VERIFIER_PROMPT = (
    "You are an AI assistant who will help me to tell if a model response "
    "matches with the groundtruth answer of a question. "
    "Semantic equivalence and reasonable paraphrases are acceptable. "
    'The decision are only Yes / No. \n'
    "If the model response correctly state the groundtruth answer, output Yes. "
    'Otherwise, output No. \n'
    'QUESTION: {}\n'
    'GROUNDTRUTH ANSWER: {}\n'
    "MODEL RESPONSE: {}; \n"
    'Please make your decision based on the above information. '
)


def model_verify_answer(question, answer, model_response):
    ak_list = [
        "EMPTY",  # TODO: add your API key here
    ]
    api_key = random.choice(ak_list)
    client = OpenAI(
        api_key=api_key,
        base_url="EMPTY",  # TODO: add your API server here
    )

    while True:
        try:
            response = client.chat.completions.create(
                model="MODEL_NAME",  # TODO: add your model name here
                messages=[
                    {
                        "role": "user",
                        "content": VERIFIER_PROMPT.format(question.replace('\n', ' '), answer, model_response),
                    }
                ],
            )
            break
        except Exception as e:
            print(e)
            time.sleep(10)

    full_resp = json.loads(response.model_dump_json())["choices"][0]["message"]["content"]
    return full_resp == "Yes"


def can_to_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def single_number_accuracy(line, response):
    if response == "":
        print(f"response is empty: {response}")
        return 0.0

    try:
        question = line['prompt']
        answer = int(line['answer'])
        response = response.split('\n')[-1]

        prediction = response.strip('* ')
        if can_to_int(prediction):
            prediction = int(prediction)
        else:
            print(f"prediction {prediction} not int")
            return model_verify_answer(question, answer, response)
    except Exception as e:
        print(f"error: {e}")
        return 0.0

    return float(answer == prediction)


def multi_choice_accuracy(line, response):
    if response == "":
        print(f"response is empty: {response}")
        return 0.0

    try:
        question = line['prompt']
        answer = line['answer']
        response = response.split('\n')[-1]
        if response.startswith('Option: '):
            response = response[len('Option: '):]
        if response.startswith('Option'):
            response = response[len('Option'):]
        if response.startswith('Answer: '):
            response = response[len('Answer: '):]

        prediction = response.strip('* ')
        if prediction not in ['A', 'B', 'C', 'D']:
            print(f"prediction {prediction} not in ['A', 'B', 'C', 'D']")
            return model_verify_answer(question, answer, response)
    except Exception as e:
        print(f"error: {e}")
        return 0.0

    return float(answer == prediction)


def multihop_accuracy(line, response):
    answer = line['answer']
    if answer in ['A', 'B', 'C', 'D']:
        return multi_choice_accuracy(line, response)
    else:
        question = line['prompt']
        answer = line['answer']
        response = response.split('\n')[-1]
        return float(model_verify_answer(question, answer, response))


def maze_accuracy(line, response):
    score, drawed_gray_array = wall_judge(response, line['image'][0], line['answer'], maze_size="5")
    return score == 1.0, drawed_gray_array


def sokoban_accuracy(line, response):
    return sokoban_compute_reward(response, None, json.loads(line['extra_info']['board'])) == 1.0


def remove_boxed(s: str) -> str:
    left = "\\boxed{"

    assert s[:len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"

    return s[len(left):-1]


def last_boxed_only_string(string: str):
    """
    find last \\boxed{...}
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
        retval = remove_boxed(retval)

    return retval


def boxed_accuracy(line, response):
    try:
        answer = str(line['answer'])
        prediction = last_boxed_only_string(response)
    except:
        return 0.0

    return float(answer == prediction)
