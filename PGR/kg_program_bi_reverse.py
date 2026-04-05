import re
import json
import requests
import time
import ast


def chat_responce(model_name='qwen/qwen3.5-122b-a10b', messages=[], system_prompt=[{'role': 'system', 'content': 'You are the powerful GPT-4 created by OpenAI.'}], max_tokens=2048, temperature=0.7, top_p=0.1):
    url = 'https://integrate.api.nvidia.com/v1/chat/completions'
    headers = {"Content-Type": "application/json", "Authorization": "Bearer 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"}
    
    messages = system_prompt + messages
    data = {"model":model_name, "messages":messages, "stream": False, "max_tokens": max_tokens, "temperature": temperature, "top_p":top_p, "chat_template_kwargs": {"enable_thinking":False}}

    while True:
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=120)
            resp_json = resp.json()
            if 'error' in resp_json:
                raise ValueError(f"API error: {resp_json['error']}")
            content = resp_json['choices'][0]['message']['content']
            if content is None:
                content = ""
            time.sleep(1.6)  # Tránh Rate Limit 40 RPM
            return content.strip()
        except Exception as e:
            print(f"[CẢNH BÁO LỖI API] Đang treo 10s để hệ thống NVIDIA nhả Limit: {e}")
            time.sleep(10)
    

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def extract_json_from_string(input_string):
    def is_json(my_str: str) -> bool:
        try:
            json.loads(my_str)
            return True
        except (ValueError, TypeError):
            return False
    
    match = re.search(r'```json\n(.*?)```', input_string, re.DOTALL)
    if match:
        return ast.literal_eval(match.group(1).strip())
    else:
        if(is_json(input_string)):
            return json.loads(input_string)
    return []
        
def get_relation_match_request(model_name: str, prompt_w_entity: str, qid: int, code_id: int, claim: str, tri_list: list, tgt_list: list, max_tokens: int):

    if(len(tri_list) == 0 or len(prompt_w_entity) > 8192):
        return []
    # result generation
    while True:
        try:
            program_gen_result = chat_responce(model_name=model_name, messages=[{
                        "role": "user",
                        "content": prompt_w_entity,
                    }], system_prompt=[{"role": "system", "content": "You are a helpful assistant."}], max_tokens=max_tokens, temperature=0.2, top_p=0.1)
            result_list = extract_json_from_string(program_gen_result)
            program_gen_dict = {'claim':claim, 'triplets':tri_list, 'matched':result_list, 'generate_string': program_gen_result}
            break
        except Exception as e:
            print("[ERROR]", e)
            time.sleep(5)
    
    with open(f'./result/relation_matched/{qid}_{code_id}_relation.json', 'w') as f:
        json.dump(program_gen_dict, f, indent=4)

    return result_list

def match_relation(model_name, triplet_set, claim: str, tgt_list: list, qid, code_id): # for SEARCH
    match_list = []

    prompt_w_entity = open_file('./prompts/search_direct_prompt.txt').replace('<<<<LIST>>>>', str(triplet_set)).replace('<<<<CLAIM>>>>', claim).replace('<<<<TARGET>>>>', str(tgt_list))

    match_list = get_relation_match_request(model_name, prompt_w_entity, qid, code_id, claim, triplet_set, tgt_list, max_tokens=1024)
    
    entity_list = []
    relation_list = [w[1] for w in match_list]
    if(tgt_list[0] == "Unknown"):
        entity_list = [w[0] for w in match_list]
    else:
        entity_list = [w[2] for w in match_list]

    return relation_list

def match_triplet(model_name, triplet_set, claim: str, tgt_list: list, qid, code_id): # for MATCH
    prompt_w_entity = open_file('./prompts/match_direct_prompt.txt').replace('<<<<LIST>>>>', str(triplet_set)).replace('<<<<CLAIM>>>>', claim).replace('<<<<TARGET>>>>', str(tgt_list))

    match_list = get_relation_match_request(model_name, prompt_w_entity, qid, code_id, claim, triplet_set, tgt_list, max_tokens=1024)

    return match_list

def SEARCH(model_name, triplet, graph, claim, qid, code_id):
    # identifies the missing entity and returns possible entities
    head, relation, tail = triplet
    possible_entities = []

    r_match = ''
    if head is None:
        if(tail not in graph):
            return []
        if('~' + relation in graph[tail]):
            r_match = '~' + relation
            possible_entities.extend(graph[tail][r_match])
        else:
            triplet_set_0 = [[graph[tail][rel][0], rel[1:], tail] for rel in graph[tail] if rel[0] == '~'] + [[tail, rel, graph[tail][rel][0]] for rel in graph[tail] if rel[0] != '~']
            relation_match = match_relation(model_name, triplet_set_0, claim, tgt_list=["Unknown", relation, tail], qid=qid, code_id=code_id)
            triplet_match = []
            for rel in relation_match:
                if(rel[0] != '~' and rel in graph[tail]):
                    triplet_match.extend(graph[tail][rel])
                elif('~' + rel in graph[tail]):
                    triplet_match.extend(graph[tail]['~' + rel])
            print(f'Search triplet_match: {triplet_match}')
            possible_entities.extend(triplet_match)
    elif tail is None:
        if(head not in graph):
            return []
        if(relation in graph[head]):
            r_match = relation
            possible_entities.extend(graph[head][r_match])
        else:
            triplet_set_0 = [[head, rel, graph[head][rel][0]] for  rel in graph[head] if rel[0] != '~'] + [[graph[head][rel][0], rel[1:], head] for  rel in graph[head] if rel[0] == '~']
            relation_match = match_relation(model_name, triplet_set_0, claim, tgt_list=[head, relation, "Unknown"], qid=qid, code_id=code_id)
            triplet_match = []
            for rel in relation_match:
                if(rel in graph[head]):
                    triplet_match.extend(graph[head][rel])
                elif('~' + rel in graph[head]):
                    triplet_match.extend(graph[head]['~' + rel])
            print(f'Search triplet_match: {triplet_match}')
            possible_entities.extend(triplet_match)

    return possible_entities

def MATCH(model_name, triplet, graph, claim, qid, code_id):
    head, relation, tail = triplet
    possible_triplets = []

    if(not isinstance(head, list)):
        head = [head]
    if(not isinstance(tail, list)):
        tail = [tail]

    triplet_set = []
    for e1 in head:
        if(e1 not in graph):
            continue
        for e2 in tail:
            for r, e1_tails in graph[e1].items():
                if(e2 in e1_tails):
                    if(r[0] != '~'):
                        possible_triplets.append([e1, r, e2])
                    else:
                        possible_triplets.append([e2, r[1:], e1])
    if(relation in [tri[1] for tri in possible_triplets]):
        return True
    else:
        possible_triplets = [list(item) for item in set(tuple(x) for x in possible_triplets)] # remove repeated triplets
        triplet_match = match_triplet(model_name, possible_triplets, claim, [head, relation, tail], qid, code_id)
        print(f'Match triplet : rel={relation}; match triplet={triplet_match}; possible triplet={possible_triplets}')
        if(len(triplet_match) > 0):
            return True
    return False

def VERIFY(result):
    return result

# Function to execute the program code
def execute_program(model_name, program_code, graph, claim, qid):
    match = re.search(r"def program\(\):\n(.*)", program_code, re.DOTALL)
    if not match:
        raise ValueError("Invalid program format")

    body = match.group(1).strip()

    # Split and execute line by line
    lines = body.split("\n")
    local_vars = {}
    code_id = 0

    print(program_code)
    for line in lines:
        line = line.strip()

        if(line == '' or line.startswith('#')):
            continue

        # Match assignment statements
        assignment_match = re.match(r"(\w+) = (\w+)\((.*)\)", line)
        if assignment_match:
            var_name, func_name, params = assignment_match.groups()

            # Evaluate parameters, replacing variables with actual values
            params = eval(params, {}, local_vars)

            # Call the corresponding function
            code_id += 1
            if func_name == "SEARCH":
                local_vars[var_name] = SEARCH(model_name, params, graph, claim, qid, code_id)
            elif func_name == "MATCH":
                local_vars[var_name] = MATCH(model_name, params, graph, claim, qid, code_id)
            elif func_name == "VERIFY":
                local_vars[var_name] = VERIFY(params)
            else:
                raise ValueError(f"Unknown function: {func_name}")
        else:
            # Bỏ qua các dòng không hợp lệ (if/else, comments, ...) thay vì crash
            print(f"[SKIP dòng không hợp lệ]: {line}")
            continue

    # Return the final result
    return local_vars.get("predicted", None)

