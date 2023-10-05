import json

import openai
import time
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import tiktoken             # pip install tiktoken

import numpy as np

# Tokenizer
from utils import remove_stopwords_and_lemmatize

tokenizer = tiktoken.get_encoding("cl100k_base")
MAXTOKENSINHISTORY = 2500

# Get the number of tokens for a string, measured using tiktoken
def getTokenLength(strIn):
    tokens = tokenizer.encode(strIn)
    numTokens = len(tokens)
    return numTokens


def get_best_matched_action_using_sent_transformer(allowed_actions, query, model, device="cpu"):
    print(f"size of allowed_actions: {len(allowed_actions)}")
    if query in allowed_actions:
        return query, [(query, 1.0)]

    # First-pass: Filter the list of allowed actions to only include those that start with the same first token as the query
    # This is a heuristic to reduce the number of actions we need to run the sentence transformer on
    # firstTokenQuery = (query.split(" ")[0]).lower()
    # allowed_actions_filtered = []
    # for action in allowed_actions:
    # # Check if action starts with the same first token as the query
    # if (action.startswith(firstTokenQuery)):
    # allowed_actions_filtered.append(action)
    # print(f"size of allowed_actions_filtered: {len(allowed_actions_filtered)}")
    # if (len(allowed_actions_filtered) > 0):
    # allowed_actions = allowed_actions_filtered

    query_norm = remove_stopwords_and_lemmatize(text=query,
                                                do_stemming=True,
                                                lemmatize=True)
    query_tokens = set(query_norm.split(" "))
    allowed_actions_filtered = []
    word_sim = []
    for action in allowed_actions:
        action_norm = remove_stopwords_and_lemmatize(text=action,
                                                    do_stemming=True,
                                                    lemmatize=True)
        action_tokens = set(action_norm.split(" "))
        num_common_words = len(list((action_tokens).intersection(query_tokens)))
        word_sim.append(-1 * num_common_words)
        # if num_common_words > 0:
        #     allowed_actions_filtered.append(action)
    indices_actions_sorted_desc_word_sim = np.argsort(word_sim)
    if "cuda" in device:
        max_filtered_actions = 100000
    else:
        max_filtered_actions = 1000

    allowed_actions_filtered = [allowed_actions[ind] for ind in indices_actions_sorted_desc_word_sim[:max_filtered_actions]]
    print(f"actions_sorted_desc_word_sim: {allowed_actions_filtered[0:10]}")
    print(f"size of allowed_actions_filtered: {len(allowed_actions_filtered)}")

    if query == "turn on sink" or query == "turn on the sink":
        valid_activate_sink = "activate sink" in allowed_actions
        valid_index = -1
        num_common_words = -1
        if valid_activate_sink:
            valid_index = allowed_actions.index("activate sink")
            num_common_words = word_sim[valid_index]
        contains_activate_sink = "activate sink" in allowed_actions_filtered

        index_by_sim = np.where(indices_actions_sorted_desc_word_sim==valid_index)
        print(f"valid_actions contains: activate sink: valid_idx:{valid_index}\tnum_common_words:{num_common_words}\trank:{index_by_sim}")

        print(f"allowed_actions_filtered contains: activate sink: {contains_activate_sink}")

        # action_embedding = model.encode(['activate sink'])
        # query_embeddings = model.encode([query])
        # sim = cosine_similarity(
        #     query_embeddings,
        #     action_embedding
        # )
        # print(f"sent transformer sim: activate sink: {sim}")

    # Second pass: Use the sentence transformer to find the best-matched action
    action_list_embeddings = model.encode(allowed_actions_filtered)
    query_embeddings = model.encode([query])
    sim = cosine_similarity(
        query_embeddings,
        action_list_embeddings
    )
    max_id = np.argmax(sim)
    # max_val = np.max(sim)

    # Sort the actions by similarity score
    # First, pack the actions and similarity scores into a list of tuples
    action_sim_tuples = []
    for i in range(len(allowed_actions_filtered)):
        action_sim_tuples.append((allowed_actions_filtered[i], sim[0][i]))
    # Second, sort the list of tuples by the similarity score
    action_sim_tuples.sort(key=lambda x: x[1], reverse=True)
    # Return the top 5 tuples
    top5_action_sim_tuples = action_sim_tuples[:5]
    # Convert top5 to float so that it can be serialized to JSON
    top5_action_sim_tuples = [(x[0], float(x[1])) for x in top5_action_sim_tuples]
    print(f"top5_action_sim_tuples: {top5_action_sim_tuples}")

    if query == "turn on sink" or query == "turn on the sink":
        sim_idx = -1
        sim_val = -1
        i = -1
        for a, s in action_sim_tuples:
            i += 1
            if a == "activate sink":
                sim_idx = i
                sim_val = -1*s
        print(f"sent transformer sim: activate sink: {contains_activate_sink}\t{sim_idx}\t{sim_val}")

    print(f"size of allowed_actions_filtered: {len(allowed_actions_filtered)}")

    return allowed_actions_filtered[max_id], top5_action_sim_tuples


def run_chatgpt_query(prompt,
                      model_name="gpt-3.5-turbo", # pass "gpt4" for more recent model output
                      max_tokens=256,
                      temperature=0.0):
    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system",
                     "content": "You are an AI agent helping execute a science experiment "
                                "in a simulated environment with limited number of objects "
                                "and actions available at each step."
                     },
                    {"role": "user", "content": f"{prompt}"}
                    # This prompt currently assumes single turn interaction with ChatGPT,
                    # we can keep appending "user", "assistant" dialogue to this list
                    # to make use of its multi-turn feature
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            print(e)
            print("GPT3 error. Retrying in 10 seconds...")            
            time.sleep(2)

    return response


def get_chatgpt_sw_next_action(task,
                               current_obs,
                               current_inventory,
                               objects_set,
                               next_actions_set,
                               previous_actions=[],
                               previous_observations=[],
                               out_logs_file=None,
                               model="gpt-3.5-turbo",
                               plan=[]):
    objects_str = '\n '.join(objects_set)
    actions_str = '\n '.join(next_actions_set)

    # Hyperparameter -- the maximum tokens in the history.
    maxTokensInHistory = MAXTOKENSINHISTORY

    # Create an action history. This is the history of actions and observations that have been taken so far.
    # This is used to help the model understand what has happened so far, and what the next action should be.
    # This is built from the most-recent to least-recent, and stops when meeting a maximum token threshold.

    action_observation_history_str = ""
    # Iterate from the last to the 0th index
    for i in range(len(previous_actions) - 1, -1, -1):
        action = previous_actions[i]
        observation = previous_observations[i]

        action_observation_history_str_candidate = f"step: {i}\n action: {action}\n observation: {observation}\n\n" + action_observation_history_str

        # If we have reached the maximum number of tokens, stop.
        if getTokenLength(action_observation_history_str_candidate) > maxTokensInHistory:
            break
        else:
            action_observation_history_str = action_observation_history_str_candidate

    # Hints for how to use the actions
    actions_str = actions_str.replace("mix OBJ", "mix OBJ (here, OBJ should be the container the items to be mixed are in)")

    sw_prompt = f"I'd like you to work your way through a virtual world to complete a particular task. At each step, tell me which action you want to do, e.g., pick up something, open something, look around etc. and I will tell you the result. Then tell me the next action you want to do, until you complete the task." \
                f"\n\n" \
                f"Task: {task}" \
                f"\n\n"

    if len(plan):
        plan_str = '\n  '.join(plan)
        sw_prompt += f"Your high level plan to accomplish this task is:" \
                     f"\n\n" \
                     f"{plan_str}" \
                     f"\n\n"

    sw_prompt += f"Your previous actions and observations:" \
                f"\n" \
                f"{action_observation_history_str}" \
                f"\n"\
                f"Here is what you currently see:" \
                f"\n" \
                f" {current_obs}" \
                f"Here is what is currently in your inventory:" \
                f"\n" \
                f" {current_inventory}" \
                f"\n\n" \
                f"Possible objects ( value an OBJ can take ) :" \
                f"\n {objects_str}" \
                f"\nYour next action should be in one of the following formats:" \
                f"\nPossible actions:" \
                f"\n {actions_str}" \
                f"\n\n" \
                f"If I say \"Ambiguous request\", your action might mean multiple things.  In that case, respond with the number corresponding to the action you want to take.\n" \
                f"\n\n" \
                f"What action would you like to do next? " \
                f"Please describe your reasoning for picking a next action.  Then, write ### followed by the single next action you would like to take." \
                f"If you think you have completed the task, please write TASK_COMPLETE as the next action."
                
    print(f"ChatGPT prompt:\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n{sw_prompt}\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    response = run_chatgpt_query(prompt=sw_prompt,
                                 model_name=model,
                                 max_tokens=256,
                                 temperature=0.7, # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
                                 )
    print(f"current_state:{current_obs}")
    # print(f"=====\nNEXT ACTION \ntask:{task}\n current_state:{current_state}"
    #       f"\nobjects_set: {objects_set}\nnext_actions_set:{next_actions_set}")
    # print(f"response:{response}")
    if out_logs_file:
        out_logs_file.write(f"next action\t{prompt}\t{json.dumps(response)}")

    # Sometimes ChatGPT returns a long string with actions mentioned in ""
    # Extract strings within double quotes

    response_str = response['choices'][0]['message']['content']
    print ("RAW RESPONSE STRING:")
    print (response_str)
    print ("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    #next_actions = re.findall('"([^"]*)"', response_str)
    ## If no such quoted actions found, consider entire generation as next action
    #if not next_actions:
    #    next_actions = [response['choices'][0]['message']['content']]

    # actions should be between "###" blocks. Take the first one (in case it generated multiple ones) as the next action
    possibleActions = response_str.split("###")
    reasoningStr = " "
    next_action_str = " "
    if len(possibleActions) > 1:
        reasoningStr = possibleActions[0]
        next_action_str = possibleActions[1].lower().strip()        # The first index should be it's reasoning, the second should be it's action. 
        # It's possible it might put multiple actions in the next_action_str, that are comma delimited. Trim out all but the first one. 
        next_action_str = next_action_str.split(",")[0].lower().strip()


    #next_action_str = next_actions[0].lower().strip()
    next_action_str = next_action_str.replace(".", "").replace("i would like to ", "").split(' and ')[0]
    response['pred_next_action'] = next_action_str    

    # Check to make sure the reasoningStr and next actions are not blank (to prevent the data structure from crashing with blank strings)
    if len(reasoningStr) < 1:
        reasoningStr = " "
    if (len(next_action_str) < 1):
        next_action_str = " UNKNOWN "

    response['reasoningStr'] = reasoningStr

    # print(f"pred_next_action:{next_actions[0]}")

    return response

######################### Run ChatGPT in chat/multi-turn mode  #######################


def run_chatgpt_query_multi_turn(messages,
                      model_name="gpt-3.5-turbo",  # pass "gpt4" for more recent model output
                      max_tokens=256,
                      temperature=0.0):
    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            print(e)
            print("GPT3 error. Retrying in 10 seconds...")
            time.sleep(2)

    return response


def get_chatgpt_sw_next_action_multi_turn(task,
                                          current_obs,
                                          current_inventory,
                                          objects_set,
                                          next_actions_set,
                                          previous_messages=[],
                                          out_logs_file=None,
                                          model="gpt-3.5-turbo",
                                          plan=[]):
    next_query = ""

    new_messages = [
        {"role": "system",
         "content": "You are an AI agent helping execute a science experiment "
                    "in a simulated environment with limited number of objects "
                    "and actions available at each step."
         }
    ]

    # Always have task information as first message
    sw_prompt_task = f"I'd like you to work your way through a virtual world to complete a particular task. At each step, tell me which action you want to do, e.g., pick up something, open something, look around etc. and I will tell you the result. Then tell me the next action you want to do, until you complete the task." \
                     f"\n\n" \
                     f"Task: {task}" \
                     f"\n\n"

    if len(plan):
        plan_str = '\n  '.join(plan)
        sw_prompt_task += f"Your high level plan to accomplish this task is:" \
                          f"\n\n" \
                          f"{plan_str}" \
                          f"\n\n"

    sw_prompt_task += f"Below you can see the most recent history of you actions " \
                      f"to help you decide your next action."

    new_messages.append(
        {"role": "user",
         "content": f"{sw_prompt_task}"
         }
    )
    if len(previous_messages):

        # Hyperparameter -- the maximum tokens in the history.
        maxTokensInHistory = MAXTOKENSINHISTORY

        # Create an action history. This is the history of actions and observations that have been taken so far.
        # This is used to help the model understand what has happened so far, and what the next action should be.
        # This is built from the most-recent to least-recent, and stops when meeting a maximum token threshold.

        action_observation_history_str = ""
        start_idx = 0
        # Iterate from the last to the 0th index
        for i in range(len(previous_messages) - 1, -1, -1):
            message_rec = previous_messages[i]

            action_observation_history_str_candidate = f"{message_rec['role']} : {message_rec['content']}\n" + action_observation_history_str

            # If we have reached the maximum number of tokens, stop.
            if getTokenLength(action_observation_history_str_candidate) > maxTokensInHistory:
                start_idx = i+1
                break
            else:
                action_observation_history_str = action_observation_history_str_candidate

        new_messages_history = previous_messages[start_idx:]   # Take last k messages that fit in the token history limit
        for new_message in new_messages_history:
            new_messages.append(new_message)

    objects_str = '\n '.join(objects_set)
    actions_str = '\n '.join(next_actions_set)
    # Hints for how to use the actions
    actions_str = actions_str.replace("mix OBJ",
                                      "mix OBJ (here, OBJ should be the container the items to be mixed are in)")

    next_query += f"Here is what you currently see:" \
                 f"\n" \
                 f" {current_obs}" \
                 f"Here is what is currently in your inventory:" \
                 f"\n" \
                 f" {current_inventory}" \
                 f"\n\n" \
                 f"Possible objects ( value an OBJ can take ) :" \
                 f"\n {objects_str}" \
                 f"\nYour next action should be in one of the following formats:" \
                 f"\nPossible actions:" \
                 f"\n {actions_str}" \
                 f"\n\n" \
                 f"If I say \"Ambiguous request\", your action might mean multiple things.  In that case, respond with the number corresponding to the action you want to take.\n" \
                 f"\n\n" \
                 f"What action would you like to do next? " \
                 f"Please describe your reasoning for picking a next action. Then, write ### followed by the single next action you would like to take." \
                 f"If you think you have completed the task, please write TASK_COMPLETE as the next action."

    new_messages.append({
        "role": "user", "content": f"{next_query}"
    })

    prompt_str = [rec['role'] + ": " + rec['content'] for rec in new_messages]
    print(
        f"ChatGPT prompt:\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n{prompt_str}\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    response = run_chatgpt_query_multi_turn(
        messages=new_messages,
        model_name=model,
        max_tokens=256,
        temperature=0.7,  # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
    )
    print(f"current_state:{current_obs}")
    # print(f"=====\nNEXT ACTION \ntask:{task}\n current_state:{current_state}"
    #       f"\nobjects_set: {objects_set}\nnext_actions_set:{next_actions_set}")
    # print(f"response:{response}")
    if out_logs_file:
        out_logs_file.write(f"next action\t{prompt}\t{json.dumps(response)}")

    # Sometimes ChatGPT returns a long string with actions mentioned in ""
    # Extract strings within double quotes

    response_str = response['choices'][0]['message']['content']
    print("RAW RESPONSE STRING:")
    print(response_str)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # actions should be between "###" blocks. Take the first one (in case it generated multiple ones) as the next action
    possibleActions = response_str.split("###")
    reasoningStr = " "
    next_action_str = " "
    if len(possibleActions) > 1:
        reasoningStr = possibleActions[0]
        next_action_str = possibleActions[
            1].lower().strip()  # The first index should be it's reasoning, the second should be it's action.
        # It's possible it might put multiple actions in the next_action_str, that are comma delimited. Trim out all but the first one.
        next_action_str = next_action_str.split(",")[0].lower().strip()

    next_action_str = next_action_str.replace(".", "").replace("i would like to ", "").split(' and ')[0]
    response['pred_next_action'] = next_action_str

    # Check to make sure the reasoningStr and next actions are not blank (to prevent the data structure from crashing with blank strings)
    if len(reasoningStr) < 1:
        reasoningStr = " "
    if (len(next_action_str) < 1):
        next_action_str = " UNKNOWN "

    response['reasoningStr'] = reasoningStr

    # Append ChatGPT response and our action selection to message history
    new_messages.append({
        "role": "assistant", "content": response_str})
    new_messages.append({
        "role": "user", "content": f"Selected action: {next_action_str}"
    })

    return response, new_messages[2:]  #skip first 2 messages (system and generic task prompt)

# ABLATED Multi-turn dialogue for CLIN model : prev message history is summarized and learnings are inserted in the history
# Dont use rationale
def get_clin_sw_next_action_multi_turn_ablated(task,
                                       current_obs,
                                       current_inventory,
                                       objects_set,
                                       next_actions_set,
                                       previous_rationales=[],
                                       previous_actions=[],
                                       previous_observations=[],
                                       out_logs_file=None,
                                       model="gpt4-0613",
                                       plan=[],
                                       eval_plan=False,
                                       last_plan_step_id=-1,
                                       cheatsheet="",
                                       summary="",
                                       temperature=0.0,
                                       quadrant=1,
                                       feedback="",
                                       episodeIdx=None,
                                    ablate_grounding=False):
    previous_rationales = []
    # We first ask the model to self-reflect:
    # ** Given past action, observation history and overall plan, which plan-step are you currently at?
    # ** Emphasize current-step as current-subgoal to generate next action

    next_query = ""

    # Always have task information as first message
    sw_prompt_task = f"I'd like you to work your way through a virtual world to complete a particular task. " \
                     f"At each step, tell me which action you want to do, e.g., pick up something, " \
                     f"open something, look around etc. and I will tell you the result. " \
                     f"Then tell me the next action you want to do, until you complete the task." \
                     f"\n\n" \
                     f"Task: {task}" \
                     f"\n\n"
    if cheatsheet:
        sw_prompt_task += f"Here is a cheatsheet given to you:\n{cheatsheet}"

    if summary:
        if quadrant == 1:
            sw_prompt_task += \
                f"Here is a summary of learnings based on your previous attempts on this task." \
                f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n{summary}"
        if quadrant == 2:
            if episodeIdx == 0:
                sw_prompt_task += \
                    f"Here is a summary of learnings based on your previous attempts to solve related tasks in some environments. However, your current envionment can differ from previous environments in terms of presence of objects, starting location etc."\
                    f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. Some of these learnings can be useful for predicting your next action:\n{summary}"
            else:
                sw_prompt_task += \
                f"Here is a summary of learnings based on your previous attempts on this task." \
                f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n{summary}"

        if quadrant == 3:
            if episodeIdx == 0:
                sw_prompt_task += \
                    f"Here is a summary of learnings based on your previous attempts to some tasks in your current environment." \
                    f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. Some of these learnings can be useful for predicting your next action:\n{summary}"
            else:
                sw_prompt_task += \
                f"Here is a summary of learnings based on your previous attempts on this task." \
                f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n{summary}"

    plan_str = ""
    plan_dict = dict()
    if len(plan):
        for sid, step in enumerate(plan, start=1):
            plan_str += f"\n  ({sid}) : {step}"
            plan_dict[sid] = step

        sw_prompt_task += f"Your high level plan to accomplish this task is:" \
                          f"\n\n" \
                          f"{plan_str}" \
                          f"\n\n"

    sw_prompt_task += f"Below you can see the most recent history of you actions " \
                      f"to help you decide your next action."

    new_messages = [
        {"role": "system",
         "content": "You are an AI agent helping execute a science experiment "
                    "in a simulated environment with limited number of objects "
                    "and actions available at each step."
         },
        {"role": "user",
         "content": f"{sw_prompt_task}"
         }
    ]

    if len(previous_actions):

        # Hyperparameter -- the maximum tokens in the history.
        maxTokensInHistory = MAXTOKENSINHISTORY

        # Create an action history. This is the history of actions and observations that have been taken so far.
        # This is used to help the model understand what has happened so far, and what the next action should be.
        # This is built from the most-recent to least-recent, and stops when meeting a maximum token threshold.

        action_observation_history_str = ""
        previous_messages = []
        # Iterate from the last to the 0th index
        for i in range(len(previous_actions) - 1, max(len(previous_actions) - 6, -1) , -1): # we are only taking last three a-o pair to fit to the token length
            # rationale = previous_rationales[i]
            action = previous_actions[i]
            observation = previous_observations[i]

            # Note we prepend at the start of list
            # So we are prepending in reverse order
            # observation after ith action, selected ith action, generated rationale for ith action
            if i < len(previous_actions) - 1:
                previous_messages[:0] = [{
                    "role": "user", "content": f"{observation}\n\nWhat action would you like to do next?"
                }]
            else:
                previous_messages[:0] = [{
                    "role": "user", "content": f"{observation}"
                }]

            previous_messages[:0] = [{
                "role": "user", "content": f"Selected action: {action}"
            }]

            # previous_messages[:0] = [{
            #     "role": "assistant", "content": f"{rationale}"
            # }]

            action_observation_history_str_candidate = f"step: {i}\n action: {action}\n observation: {observation}\n\n" + action_observation_history_str
            # f"assistant: {rationale} "\
            

            # If we have reached the maximum number of tokens, stop.
            if getTokenLength(action_observation_history_str_candidate) > maxTokensInHistory:
                break
            else:
                action_observation_history_str = action_observation_history_str_candidate

        for prev_message in previous_messages:
            new_messages.append(prev_message)

    next_actions_set = [x for x in next_actions_set if 'focus' not in x]
    objects_str = '\n '.join(objects_set)
    actions_str = '\n '.join(next_actions_set)
    # Hints for how to use the actions
    actions_str = actions_str.replace("mix OBJ",
                                      "mix OBJ (here, OBJ should be the container the items to be mixed are in)")

    next_query += f"Here is what you currently see:" \
                 f"\n" \
                 f" {current_obs}" \
                 f"Here is what is currently in your inventory:" \
                 f"\n" \
                 f" {current_inventory}" \
                 f"\n\n"

    next_action_query = ''

    # Plan-grounding: After each observation, ask: What step are you currently working on?
    # Append this reflection before asking for next action
    current_plan_step_id = last_plan_step_id
    current_plan_step_rationale = ""
    if len(plan) and eval_plan:
        next_query += f"Which plan step are you currently working on? You can format the answer as (step-id) e.g. (3)"
        new_messages.append({
            "role": "user", "content": f"{next_query}"
        })
        response = run_chatgpt_query_multi_turn(
            messages=new_messages,
            model_name=model,
            max_tokens=20,
            temperature=temperature,  # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
        )
        response_str = response['choices'][0]['message']['content']
        prompt_str = [rec['role'] + ": " + rec['content'] for rec in new_messages]
        # print(f"ChatGPT plan-step prompt:\n^^^^^^^^^^^^^^^^^^^^^^^^^\n{prompt_str}\n^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print(f"response:\n^^^^^^^^^^^^^^^^^^^^^^^^^\n{response_str}\n")
        current_plan_step_rationale = response_str
        step_id_list = re.findall(r'\d+', response_str)
        if len(step_id_list):
            step_id = step_id_list[0]
            current_plan_step_id = int(step_id)
        if current_plan_step_id < 1 or current_plan_step_id > len(plan):
            current_plan_step_id = last_plan_step_id
        new_messages.append({
            "role": "assistant", "content": f"I am at step ({current_plan_step_id}) : {plan_dict[current_plan_step_id]}"
        })

        # print(f"step_id:{current_plan_step_id}\n^^^^^^^^^^^^^^^^^^^^^^^^^")

    else:
        next_action_query = next_query

    next_action_query += f"Possible objects ( value an OBJ can take ) :" \
                 f"\n {objects_str}" \
                 f"\nYour next action should be in one of the following formats:" \
                 f"\nPossible actions:" \
                 f"\n {actions_str}" \
                 f"\n\n" \
                 f"If I say \"Ambiguous request\", your action might mean multiple things. In that case, respond with the number corresponding to the action you want to take.\n" \
                 f"\n\n" \
                 f"What action would you like to do next? Single next action:"

                #  f"What action would you like to do next?\n" \
                  
    # if not ablate_grounding:
    #     next_action_query += f"If you think you have completed the task, please write TASK_COMPLETE as the next action."\
    #         f"If the task requires you to 'focus' on something (OBJ), please write FOCUS ON <OBJ> as the next action. FOCUS is a extremely critical action that can be only used the number of times 'focus' is mentioned in the task description. Using it more than that or inappropiately (such as on a wrong object) will terminate the session and the task will be rendered as incomplete." \
    #              f"If you performed an action that requires waiting to see the effect, please write 'wait' as the next action."  \
                 
    
    #  f"First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the last observation. Format your response as follows:\n" \
                #  f"Write 'I used learning id(s):' as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the rationale. "\

    if feedback:
        next_action_query += "ERROR: {}".format(feedback)
    

    new_messages.append({
        "role": "user", "content": f"{next_action_query}"
    })

    prompt_str = [rec['role'] + ": " + rec['content'] for rec in new_messages]
    # print(
        # f"ChatGPT prompt:\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n{prompt_str}\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    response = run_chatgpt_query_multi_turn(
        messages=new_messages,
        model_name=model,
        max_tokens=256,
        temperature=temperature,  # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
    )
    # print(f"current_state:{current_obs}")
    # print(f"=====\nNEXT ACTION \ntask:{task}\n current_state:{current_state}"
    #       f"\nobjects_set: {objects_set}\nnext_actions_set:{next_actions_set}")
    # print(f"response:{response}")
    if out_logs_file:
        out_logs_file.write(f"next action\t{prompt_str}\t{json.dumps(response)}")

    # Sometimes ChatGPT returns a long string with actions mentioned in ""
    # Extract strings within double quotes

    response_str = response['choices'][0]['message']['content']
    response['response_str'] = response_str
    # print("RAW RESPONSE STRING:")
    print(f"NEXT ACTION RESPONSE STR: {response_str}")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # next_actions = re.findall('"([^"]*)"', response_str)
    ## If no such quoted actions found, consider entire generation as next action
    # if not next_actions:
    #    next_actions = [response['choices'][0]['message']['content']]

    # actions should be between "###" blocks. Take the first one (in case it generated multiple ones) as the next action
    if "###" not in response_str:
        response_str = "###" + response_str
    possibleActions = response_str.split("###")
    reasoningStr = " "
    next_action_str = " "
    if len(possibleActions) > 1:
        reasoningStr = possibleActions[0]
        next_action_str = possibleActions[1].lower().strip()  # The first index should be it's reasoning, the second should be it's action.
        # It's possible it might put multiple actions in the next_action_str, that are comma delimited. Trim out all but the first one.
        # next_action_str = next_action_str.split(",")[0].lower().strip()

    # next_action_str = next_actions[0].lower().strip()
    next_action_str = next_action_str.replace(".", "").replace("i would like to ", "") # .split(' and ')[0]
    next_action_str = next_action_str.replace("selected action:", "").strip()
    response['pred_next_action'] = next_action_str if next_action_str else " "

    # Check to make sure the reasoningStr and next actions are not blank (to prevent the data structure from crashing with blank strings)
    if len(reasoningStr) < 1:
        reasoningStr = " "
    if (len(next_action_str) < 1):
        next_action_str = " UNKNOWN "

    response['reasoningStr'] = reasoningStr

    # Append ChatGPT response and our action selection to message history
    new_messages.append({
        "role": "assistant", "content": response_str})
    new_messages.append({
        "role": "user", "content": f"Selected action: {next_action_str}"
    })
    # print(f"pred_next_action:{next_actions[0]}")

    return response, current_plan_step_id, current_plan_step_rationale


def get_chatgpt_sw_plan(task, out_logs_file=None, model="gpt-3.5-turbo",
                        reflection_str="", cheatsheet="", temperature=0.7):
    sw_prompt = f"I'd like you to work your way through a virtual world to " \
                f"complete a particular task. At each step, tell me which " \
                f"action you want to do, e.g., pick up something, open something, " \
                f"look around etc. and I will tell you the result. " \
                f"Then tell me the next action you want to do, until you complete the task." \
                f"\n\n" \
                f"{task}"
    if reflection_str:
        sw_prompt += f"Your reflection from past is:\n{reflection_str}"

    if cheatsheet:
        sw_prompt += f"Here is a cheatsheet given to you:\n{cheatsheet}"

    sw_prompt += f"\n\nGenerate a high-level plan to complete this task."
    response = run_chatgpt_query(prompt=sw_prompt,
                                 model_name=model,
                                 max_tokens=512,
                                 temperature=temperature,  # this way we can sample multiple plans, set to 0 for greedy decoding
                                )
    print(f"=====\nHIGH LEVEL PLAN prompt:{task}")
    print(f"response:{response}")
    if out_logs_file:
        out_logs_file.write(f"plan\t{sw_prompt}\t{json.dumps(response)}")

    return response


# Multi-turn dialogue for CLIN model : prev message history is summarized and learnings are inserted in the history
def get_clin_sw_next_action_multi_turn(task,
                                       current_obs,
                                       current_inventory,
                                       objects_set,
                                       next_actions_set,
                                       previous_rationales=[],
                                       previous_actions=[],
                                       previous_observations=[],
                                       out_logs_file=None,
                                       model="gpt4-0613",
                                       plan=[],
                                       eval_plan=False,
                                       last_plan_step_id=-1,
                                       cheatsheet="",
                                       summary="",
                                       temperature=0.0,
                                       quadrant=1,
                                       feedback="",
                                       episodeIdx=None):
    # We first ask the model to self-reflect:
    # ** Given past action, observation history and overall plan, which plan-step are you currently at?
    # ** Emphasize current-step as current-subgoal to generate next action

    next_query = ""

    # Always have task information as first message
    sw_prompt_task = f"I'd like you to work your way through a virtual world to complete a particular task. " \
                     f"At each step, tell me which action you want to do, e.g., pick up something, " \
                     f"open something, look around etc. and I will tell you the result. " \
                     f"Then tell me the next action you want to do, until you complete the task." \
                     f"\n\n" \
                     f"Task: {task}" \
                     f"\n\n"
    if cheatsheet:
        sw_prompt_task += f"Here is a cheatsheet given to you:\n{cheatsheet}"

    if summary:
        if quadrant == 1:
            sw_prompt_task += \
                f"Here is a summary of learnings based on your previous attempts on this task." \
                f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n{summary}"
        if quadrant == 2:
            if episodeIdx == 0:
                sw_prompt_task += \
                    f"Here is a summary of learnings based on your previous attempts to solve related tasks in some environments. However, your current envionment can differ from previous environments in terms of presence of objects, starting location etc."\
                    f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. Some of these learnings can be useful for predicting your next action:\n{summary}"
            else:
                sw_prompt_task += \
                f"Here is a summary of learnings based on your previous attempts on this task." \
                f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n{summary}"

        if quadrant == 3:
            if episodeIdx == 0:
                sw_prompt_task += \
                    f"Here is a summary of learnings based on your previous attempts to some tasks in your current environment." \
                    f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. Some of these learnings can be useful for predicting your next action:\n{summary}"
            else:
                sw_prompt_task += \
                f"Here is a summary of learnings based on your previous attempts on this task." \
                f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n{summary}"

    plan_str = ""
    plan_dict = dict()
    if len(plan):
        for sid, step in enumerate(plan, start=1):
            plan_str += f"\n  ({sid}) : {step}"
            plan_dict[sid] = step

        sw_prompt_task += f"Your high level plan to accomplish this task is:" \
                          f"\n\n" \
                          f"{plan_str}" \
                          f"\n\n"

    sw_prompt_task += f"Below you can see the most recent history of you actions " \
                      f"to help you decide your next action."

    new_messages = [
        {"role": "system",
         "content": "You are an AI agent helping execute a science experiment "
                    "in a simulated environment with limited number of objects "
                    "and actions available at each step."
         },
        {"role": "user",
         "content": f"{sw_prompt_task}"
         }
    ]

    if len(previous_actions):

        # Hyperparameter -- the maximum tokens in the history.
        maxTokensInHistory = MAXTOKENSINHISTORY

        # Create an action history. This is the history of actions and observations that have been taken so far.
        # This is used to help the model understand what has happened so far, and what the next action should be.
        # This is built from the most-recent to least-recent, and stops when meeting a maximum token threshold.

        action_observation_history_str = ""
        previous_messages = []
        # Iterate from the last to the 0th index
        for i in range(len(previous_actions) - 1, max(len(previous_actions) - 6, -1) , -1): # we are only taking last three a-o pair to fit to the token length
            rationale = previous_rationales[i]
            action = previous_actions[i]
            observation = previous_observations[i]

            # Note we prepend at the start of list
            # So we are prepending in reverse order
            # observation after ith action, selected ith action, generated rationale for ith action
            if i < len(previous_actions) - 1:
                previous_messages[:0] = [{
                    "role": "user", "content": f"{observation}\n\nWhat action would you like to do next?"
                }]
            else:
                previous_messages[:0] = [{
                    "role": "user", "content": f"{observation}"
                }]

            previous_messages[:0] = [{
                "role": "user", "content": f"Selected action: {action}"
            }]

            previous_messages[:0] = [{
                "role": "assistant", "content": f"{rationale}"
            }]

            action_observation_history_str_candidate = f"assistant: {rationale} step: {i}\n action: {action}\n observation: {observation}\n\n" + action_observation_history_str

            # If we have reached the maximum number of tokens, stop.
            if getTokenLength(action_observation_history_str_candidate) > maxTokensInHistory:
                break
            else:
                action_observation_history_str = action_observation_history_str_candidate

        for prev_message in previous_messages:
            new_messages.append(prev_message)

    next_actions_set = [x for x in next_actions_set if 'focus' not in x]
    objects_str = '\n '.join(objects_set)
    actions_str = '\n '.join(next_actions_set)
    # Hints for how to use the actions
    actions_str = actions_str.replace("mix OBJ",
                                      "mix OBJ (here, OBJ should be the container the items to be mixed are in)")

    next_query += f"Here is what you currently see:" \
                 f"\n" \
                 f" {current_obs}" \
                 f"Here is what is currently in your inventory:" \
                 f"\n" \
                 f" {current_inventory}" \
                 f"\n\n"

    next_action_query = ''

    # Plan-grounding: After each observation, ask: What step are you currently working on?
    # Append this reflection before asking for next action
    current_plan_step_id = last_plan_step_id
    current_plan_step_rationale = ""
    if len(plan) and eval_plan:
        next_query += f"Which plan step are you currently working on? You can format the answer as (step-id) e.g. (3)"
        new_messages.append({
            "role": "user", "content": f"{next_query}"
        })
        response = run_chatgpt_query_multi_turn(
            messages=new_messages,
            model_name=model,
            max_tokens=20,
            temperature=temperature,  # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
        )
        response_str = response['choices'][0]['message']['content']
        prompt_str = [rec['role'] + ": " + rec['content'] for rec in new_messages]
        # print(f"ChatGPT plan-step prompt:\n^^^^^^^^^^^^^^^^^^^^^^^^^\n{prompt_str}\n^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print(f"response:\n^^^^^^^^^^^^^^^^^^^^^^^^^\n{response_str}\n")
        current_plan_step_rationale = response_str
        step_id_list = re.findall(r'\d+', response_str)
        if len(step_id_list):
            step_id = step_id_list[0]
            current_plan_step_id = int(step_id)
        if current_plan_step_id < 1 or current_plan_step_id > len(plan):
            current_plan_step_id = last_plan_step_id
        new_messages.append({
            "role": "assistant", "content": f"I am at step ({current_plan_step_id}) : {plan_dict[current_plan_step_id]}"
        })

        # print(f"step_id:{current_plan_step_id}\n^^^^^^^^^^^^^^^^^^^^^^^^^")

    else:
        next_action_query = next_query

    next_action_query += f"Possible objects ( value an OBJ can take ) :" \
                 f"\n {objects_str}" \
                 f"\nYour next action should be in one of the following formats:" \
                 f"\nPossible actions:" \
                 f"\n {actions_str}" \
                 f"\n\n" \
                 f"If I say \"Ambiguous request\", your action might mean multiple things. In that case, respond with the number corresponding to the action you want to take.\n" \
                 f"\n\n" \
                 f"What action would you like to do next?\n" \
                 f"First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the last observation. Format your response as follows:\n" \
                 f"Write 'I used learning id(s):' as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the rationale. Finally, write ### followed by the single next action you would like to take." \
                 f"If you think you have completed the task, please write TASK_COMPLETE as the next action." \
                 f"If the task requires you to 'focus' on something (OBJ), please write FOCUS ON <OBJ> as the next action. FOCUS is a extremely critical action that can be only used the number of times 'focus' is mentioned in the task description. Using it more than that or inappropiately (such as on a wrong object) will terminate the session and the task will be rendered as incomplete." \
                 f"If you performed an action that requires waiting to see the effect, please write 'wait' as the next action."  \
    
    if feedback:
        next_action_query += "ERROR: {}".format(feedback)
    

    new_messages.append({
        "role": "user", "content": f"{next_action_query}"
    })

    prompt_str = [rec['role'] + ": " + rec['content'] for rec in new_messages]
    # print(
        # f"ChatGPT prompt:\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n{prompt_str}\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    response = run_chatgpt_query_multi_turn(
        messages=new_messages,
        model_name=model,
        max_tokens=256,
        temperature=temperature,  # 0 for greedy best decoding       # PJ modified from 0.7 to 0.0
    )
    # print(f"current_state:{current_obs}")
    # print(f"=====\nNEXT ACTION \ntask:{task}\n current_state:{current_state}"
    #       f"\nobjects_set: {objects_set}\nnext_actions_set:{next_actions_set}")
    # print(f"response:{response}")
    if out_logs_file:
        out_logs_file.write(f"next action\t{prompt_str}\t{json.dumps(response)}")

    # Sometimes ChatGPT returns a long string with actions mentioned in ""
    # Extract strings within double quotes

    response_str = response['choices'][0]['message']['content']
    response['response_str'] = response_str
    # print("RAW RESPONSE STRING:")
    # print(response_str)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # next_actions = re.findall('"([^"]*)"', response_str)
    ## If no such quoted actions found, consider entire generation as next action
    # if not next_actions:
    #    next_actions = [response['choices'][0]['message']['content']]

    # actions should be between "###" blocks. Take the first one (in case it generated multiple ones) as the next action
    possibleActions = response_str.split("###")
    reasoningStr = " "
    next_action_str = " "
    if len(possibleActions) > 1:
        reasoningStr = possibleActions[0]
        next_action_str = possibleActions[1].lower().strip()  # The first index should be it's reasoning, the second should be it's action.
        # It's possible it might put multiple actions in the next_action_str, that are comma delimited. Trim out all but the first one.
        # next_action_str = next_action_str.split(",")[0].lower().strip()

    # next_action_str = next_actions[0].lower().strip()
    next_action_str = next_action_str.replace(".", "").replace("i would like to ", "") # .split(' and ')[0]
    response['pred_next_action'] = next_action_str

    # Check to make sure the reasoningStr and next actions are not blank (to prevent the data structure from crashing with blank strings)
    if len(reasoningStr) < 1:
        reasoningStr = " "
    if (len(next_action_str) < 1):
        next_action_str = " UNKNOWN "

    response['reasoningStr'] = reasoningStr

    # Append ChatGPT response and our action selection to message history
    new_messages.append({
        "role": "assistant", "content": response_str})
    new_messages.append({
        "role": "user", "content": f"Selected action: {next_action_str}"
    })
    # print(f"pred_next_action:{next_actions[0]}")

    return response, current_plan_step_id, current_plan_step_rationale


def get_chatgpt_sw_plan(task, out_logs_file=None, model="gpt-3.5-turbo",
                        reflection_str="", cheatsheet="", temperature=0.7):
    sw_prompt = f"I'd like you to work your way through a virtual world to " \
                f"complete a particular task. At each step, tell me which " \
                f"action you want to do, e.g., pick up something, open something, " \
                f"look around etc. and I will tell you the result. " \
                f"Then tell me the next action you want to do, until you complete the task." \
                f"\n\n" \
                f"{task}"
    if reflection_str:
        sw_prompt += f"Your reflection from past is:\n{reflection_str}"

    if cheatsheet:
        sw_prompt += f"Here is a cheatsheet given to you:\n{cheatsheet}"

    sw_prompt += f"\n\nGenerate a high-level plan to complete this task."
    response = run_chatgpt_query(prompt=sw_prompt,
                                 model_name=model,
                                 max_tokens=512,
                                 temperature=temperature,  # this way we can sample multiple plans, set to 0 for greedy decoding
                                )
    print(f"=====\nHIGH LEVEL PLAN prompt:{task}")
    print(f"response:{response}")
    if out_logs_file:
        out_logs_file.write(f"plan\t{sw_prompt}\t{json.dumps(response)}")

    return response


def get_chatgpt_sw_plan_reflection(task, generated_plan, action_seq, final_score, out_logs_file=None,
                                   model="gpt-3.5-turbo", temperature=0.7):
    plan_str = ""
    plan_dict = dict()
    if len(generated_plan):
        for sid, step in enumerate(generated_plan, start=1):
            plan_str += f"\n  ({sid}) : {step}"
            plan_dict[sid] = step

    action_str = ""
    if len(action_seq):
        for sid, step in enumerate(action_seq, start=1):
            action_str += f"\n {step} "

    messages = []
    reflection_prompt = \
        f"I'd like you to work your way through a virtual world to " \
        f"complete a particular TASK. " \
        f"You have attempted this task before. " \
        f"I would like you to to read your PLAN, your ACTION-SEQUENCE and the REWARD you got for the task." \
        f"and record your reflections in natural language." \
        f"You can use these reflections to improve your performance in the next attempt at this task." \
        f"\n\n" \
        f"TASK:{task}\n\n" \
        f"Your previous PLAN: \n{plan_str}\n\n"\
        f"Your previous ACTION-SEQUENCE:\n {action_str}\n\n" \
        f"After executing this action sequence you got REWARD: {final_score} out of 100.\n\n" \
        f"\nPlease reflect on the above plan and your proposed action sequence."\
        f"\nWhat would you like to tell your future self so that you succeed at this task "\
        f"using minimum number of actions."

    messages.append({"role": "user", "content": reflection_prompt})
    if out_logs_file:
        out_logs_file.write(f"plan\t{task_prompt}\t{json.dumps(response)}")
    response = run_chatgpt_query_multi_turn(
        messages=messages,
        model_name=model,
        max_tokens=256,
        temperature=temperature,  # 0 for greedy best decoding
    )
    print(f"=====\nREFLECTION prompt:{messages}")
    response_str = response['choices'][0]['message']['content']
    print(f"response:{response_str}")
    return response_str


def get_plan_from_chatgpt(task, model, reflection_str="", cheatsheet="", temperature=0.7):
    response = get_chatgpt_sw_plan(task=task, model=model, reflection_str=reflection_str, cheatsheet=cheatsheet, temperature=temperature)
    plan_str = response['choices'][0]['message']['content'].lower()

    plan_str = "1." + plan_str.split('1.')[1]
    plan = []
    for action in plan_str.split('\n'):
        if len(action.split('.')) > 1:
            plan_step = (''.join(action.split('.')[1:-1])).strip()  # remove integer indices at the start of plan step
            if plan_step:   # ignore empty plan steps
                plan.append(plan_step)
    return plan


def get_chatgpt_sw_plan_reflection_q2(task, generated_plan, action_seq, final_score, out_logs_file=None,
                                       model="gpt-3.5-turbo", temperature=0.7):
        plan_str = ""
        plan_dict = dict()
        if len(generated_plan):
            for sid, step in enumerate(generated_plan, start=1):
                plan_str += f"\n  ({sid}) : {step}"
                plan_dict[sid] = step

        action_str = ""
        if len(action_seq):
            for sid, step in enumerate(action_seq, start=1):
                action_str += f"\n {step} "

        messages = []
        reflection_prompt = \
            f"I'd like you to work your way through a virtual world to " \
            f"complete a particular TASK. " \
            f"You have attempted this task before. " \
            f"I would like you to to read your PLAN, your ACTION-SEQUENCE and the REWARD you got for the task." \
            f"and record your reflections in natural language." \
            f"You can use these reflections to attempt the same task in a different environment." \
            f"\n\n" \
            f"TASK:{task}\n\n" \
            f"Your previous PLAN: \n{plan_str}\n\n" \
            f"Your previous ACTION-SEQUENCE:\n {action_str}\n\n" \
            f"After executing this action sequence you got REWARD: {final_score} out of 100.\n\n" \
            f"\nPlease reflect on the above plan and your proposed action sequence." \
            f"\nWhat would you like to tell your future self so that you succeed at this task " \
            f"in a different environment using minimum number of actions."

        messages.append({"role": "user", "content": reflection_prompt})
        if out_logs_file:
            out_logs_file.write(f"plan\t{task_prompt}\t{json.dumps(response)}")
        response = run_chatgpt_query_multi_turn(
            messages=messages,
            model_name=model,
            max_tokens=256,
            temperature=temperature,  # 0 for greedy best decoding
        )
        print(f"=====\nREFLECTION prompt:{messages}")
        response_str = response['choices'][0]['message']['content']
        print(f"response:{response_str}")
        return response_str


        # Example python call for Next Action
# -------------------------------------
# res = get_chatgpt_sw_next_action(task="Your task is to grow a apple plant from seed. Seeds can be found in the kitchen. First, focus on a seed. Then, make changes to the environment that grow the plant until it reaches the reproduction life stage.", current_state="This room is called the hallway. In it, you see: table, chair, door to kitchen", next_actions_set=['go', 'search', 'dig'])
#
# Example output:
# response:{
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "message": {
#         "content": "I would like to go to the kitchen.",
#         "role": "assistant"
#       }
#     }
#   ],
#   "created": 1679337581,
#   "id": "chatcmpl-6wEhx6SSIx61CdFCvzugd5NpecPdS",
#   "model": "gpt-3.5-turbo-0301",
#   "object": "chat.completion",
#   "usage": {
#     "completion_tokens": 10,
#     "prompt_tokens": 204,
#     "total_tokens": 214
#   }
# }

# Example python call for High Level Plan
# ----------------------------------------
# res = get_chatgpt_sw_plan(task="Your task is to grow a apple plant from seed. Seeds can be found in the kitchen. First, focus on a seed. Then, make changes to the environment that grow the plant until it reaches the reproduction life stage.")
#
# Example output:
# response:{
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "message": {
#         "content": "Sure, here's a high-level plan to complete the task:\n\n1. Look around the simulated environment to locate the kitchen.\n2. Navigate to the kitchen and search for a seed.\n3. Once the seed is found, pick it up and bring it to a suitable location for planting.\n4. Plant the seed in the soil or a suitable growing medium.\n5. Water the seed regularly to ensure it has enough moisture to germinate and grow.\n6. Provide adequate light for the seed to grow.\n7. Monitor the plant's growth and provide appropriate nutrients as necessary.\n8. Continue to care for the plant until it reaches the reproduction life stage.\n\nDoes this plan work for you?",
#         "role": "assistant"
#       }
#     }
#   ],
#   "created": 1679337732,
#   "id": "chatcmpl-6wEkOGhANuzLgBhosZUVMPGTbNL3t",
#   "model": "gpt-3.5-turbo-0301",
#   "object": "chat.completion",
#   "usage": {
#     "completion_tokens": 139,
#     "prompt_tokens": 163,
#     "total_tokens": 302
#   }
# }

def success_map(metric, score):
    feedback = ''
    if metric == 'reward':
        if score == -100:
            feedback += "The agent made critical mistakes and the task was terminated."
        if score < 0:
            feedback += "The agent performed very poorly and could not make any critical progress."
        if score >= 0 and score < 20:
            feedback += "The agent performed poorly and made some progress but not enough to solve the task."
        if score >= 20 and score < 50:
            feedback += "The agent performed moderately and made some critical progress but not enough to solve the task."
        if score >= 50 and score < 90:
            feedback += "The agent performed very well and made significant critical progress but not enough to solve the task."
        if score >= 90 and score < 100:
            feedback += "The agent performed exceptionally well and made significant critical progress, was just slight away from solving the task."
        if score == 100:
            feedback += "The agent performed exceptionally well and successfully solved the task."

    if metric == 'precision_final':
        if score >= 0 and score < 10:
            feedback += "The final state of the world reached by the agent did not overlap with the desired final state of the world to solve the task, indicating low quality of final states."
        if score >= 10 and score < 50:
            feedback += "The final state of the world reached by the agent somewhat overlapped with the desired final state of the world to solve the task, indicating moderate low quality of final states."
        if score > 50 and score < 80:
            feedback += "The final state of the world reached by the agent significantly overlapped with the desired final state of the world to solve the task, indicating moderate high quality of final states."
        if score >= 80 and score <= 100:
            feedback += "The final state of the world reached by the agent overlapped almost fully with the desired final state of the world to solve the task, indicaiting high quality of final states."
    
    if metric == 'recall_final':
        if score >= 0 and score < 10:
            feedback += "A low fraction of the desired final states of the world to solve the task were achieved by the agent."
        if score >= 10 and score < 50:
            feedback += "A moderate low fraction of the desired final states of the world to solve the task were achieved by the agent."
        if score > 50 and score < 80:
            feedback += "A moderate high fraction of the desired final states of the world to solve the task were achieved by the agent."
        if score >= 80 and score <= 100:
            feedback += "A high fraction of the desired final states of the world to solve the task were achieved by the agent."
    
    return feedback

def get_trace(data, truncate=False, quadrant=1):
    trace = "\n\nCURRENT TRACE\n\n"
    trace += "Task: {}\n\n".format(data["taskDescription"])
    if data['history']:
        for item in data['history']:
            # print(item)
            # trace += "Rationale: {}\n".format(item.get('rationale', ""))
            trace += "Action: {}\n".format(item['action'])
            if truncate:
                trace += "Observation: {}\n\n".format(item['observation'].split('.')[0])
            else:
                trace += "Observation: {}\n\n".format(item['observation'])

            # optional cummulative PR

        # we assume if PRF is not computed, we will not have this field
        trace += "\n\nEVALUATION REPORT:\n"
        if 'state_change_scores' in data and quadrant != 0:
            if 'final_state' in data['state_change_scores']:
                trace += "PRECISION_FINAL: {}. This means: {}\n".format(data['state_change_scores']['final_state']['precision'], success_map('precision_final', data['state_change_scores']['final_state']['precision']))
                trace += "RECALL_FINAL: {}. This means: {}\n".format(data['state_change_scores']['final_state']['recall'], success_map('recall_final', data['state_change_scores']['final_state']['recall']))
        trace += "REWARD_FINAL: {}. This means: {}\n".format(data['finalScore'], success_map('reward', data['finalScore']))

    return trace

def format_memory(memories):
    # memories list of last-k jsons
    memory_string = "\n\nPREVIOUS LEARNINGS\n\n"
    for m in memories:
        if m['summary']:
            memory_string += "TASK: {}\n".format(m['taskDescription'])
            memory_string += "EPISODE: {}\n".format(m['episodeIdx'])
            memory_string += "LEARNINGS: {}\n".format(m['summary'])

            memory_string += "\nEVALUATION REPORT (for the attempt associated with the learning):\n"
            if 'state_change_scores' in m:
                if 'final_state' in m['state_change_scores']:
                    # memory_string += "PRECISION_FINAL: {}\n".format(m['state_change_scores']['final_state']['precision'])
                    # memory_string += "RECALL_FINAL: {}\n".format(m['state_change_scores']['final_state']['recall'])
                    p_final = m['state_change_scores']['final_state']['precision']
                    r_final = m['state_change_scores']['final_state']['recall']
                    p_all = m['state_change_scores']['all_state_changes']['precision']
                    r_all = m['state_change_scores']['all_state_changes']['recall']

                            
                    memory_string += "PRECISION_FINAL: {}. This means: {}\n".format(p_final, success_map('precision_final', p_final))
                    memory_string += "RECALL_FINAL: {}. This means: {}\n".format(r_final, success_map('recall_final', r_final))

                    # if p_final < 100 or r_final < 100 or p_all < 100 or r_all < 100:
                    #     memory_string += "The agent deviated from the desired trajectory to complete this task.\n"
                    # else:
                    #     memory_string += "The agent discovered the desired trajectory to complete this task.\n"

                    # memory_string += "PRECISION_FINAL: {}\n".format(m['state_change_scores']['final_state']['precision'])
                    # memory_string += "RECALL_FINAL: {}\n".format(m['state_change_scores']['final_state']['recall'])

                    # memory_string += f"Agent's actions resulted in {p_all}% accurate state changes and " \
                                    #  f"covered {r_all}% of desired state changes to the environment.\n"
                    # memory_string += f"Agent's actions resulted in certain state changes to the environment." \
                    #                  f"{p_all}% of those were necessary and they covered {r_all}% of desired or gold state changes.\n"
                    
                    # memory_string += f"The agent achieved a final state that was {p_final}% accurate and " \
                                    #  f"covered {r_final}% of desired final state.\n"
                    # memory_string += f"The agent achieved {r_final}% of the desired final state."

            final_score = m['finalScore']
            memory_string += "REWARD_FINAL: {}. This means: {}\n".format(final_score, success_map('reward', final_score))
            # memory_string += "REWARD_FINAL: {}\n".format()
            memory_string += '\n'
    
    return memory_string

def summarize(trace, summary_prompt, system_prompt, demo_examples="", prev_memories="", model="gpt4", temp=0.7, tokens=1000):
    print(f"trace:{trace}")
    print(f"summary_prompt:{summary_prompt}")
    print(f"system_prompt:{system_prompt}")
    print(f"demo_examples:{demo_examples}")
    print(f"prev_memories:{prev_memories}")

    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": summary_prompt},
                            {"role": "user", "content": demo_examples},
                            {"role": "user", "content": prev_memories},
                            {"role": "user", "content": trace}],
                    temperature=temp, 
                    max_tokens=tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
        except Exception as e:
            print(e)
            print("GPT3 error. Retrying in 10 seconds...")
            time.sleep(2)

    output_summary = response["choices"][0]["message"]["content"]
    return output_summary

def summarize_trace_for_preconditions_sTsW(current_run,
                        prev_runs_list=None,
                        gold_run=None,
                        demo_examples=None,
                        model="gpt-4-0613",
                        temp=0,
                        quadrant=1,
                        meta_summary=None,
                        use_last_k_memories=3):

    if gold_run:
        return gold_run['summary'] # returns gold memory

    prev_memories = ""
    meta_summary_prompt = ''
    # meta_summary_prompt_q3 = ''
    
    if len(prev_runs_list) < use_last_k_memories:
        if quadrant == 2:
            if meta_summary:
                meta_summary_prompt = f"You are also provided with a set of META LEARNINGS that contains useful insights from agent's previous best attempts to solve a same type of tasks that the you are currently solving in different environment configurations. Previous environment configurations may differ from the current one you are in, in terms of presence of objects, starting location etc."
                prev_memories = "META LEARNINGS:\n" + meta_summary + prev_memories
        if quadrant == 3:
            if meta_summary:
                meta_summary_prompt = f"You are also provided with a set of META LEARNINGS that contains useful insights from agent's previous best attempts to solve tasks in the SAME environment configuration that you are currently in. These learnings will contain related information about the environment such as presence of objects, starting location, navigational information, etc."
                prev_memories = "META LEARNINGS:\n" + meta_summary + prev_memories

    system_prompt = "You are an expert assistant."
    summary_prompt = "You are given CURRENT TRACE, a sequence of actions that an agent made in a world to accomplish a task." \
                            "Task is detailed at the beginning." \
                            "For each action, there is a rationale why the agent made that action.\n" \
                            "There is an observation that provide details about the new state of the world after each action was executed." \
                            "The CURRENT TRACE is accompanied by an EVALUATION REPORT indicating the success of the attempt to the task.\n\n" \
                            "You can also be provided with PREVIOUS LEARNINGS which are learnings from the previous attempts by the agent for the same task in the same environment/world. TASK indicates the task description. EPISODE indicates the number of previous attempts of the task.\n" \
                            "PREVIOUS LEARNINGS also have EVALUATION REPORTs indicated how sucessful the respective attempt was for solving the task."
    
    if quadrant != 0:
        task_prompt = "Generate a summary of learning, as a numbered list, that will help the agent to successfully accomplish the SAME task AGAIN, in the SAME world.\n" \
                "Each numbered item in the summary can ONLY be of the form:\n" \
                            "X MAY BE NECCESSARY to Y.\n" \
                            "X SHOULD BE NECCESSARY to Y.\n" \
                            "X MAY BE CONTRIBUTE to Y.\n" \
                            "X DOES NOT CONTRIBUTE to Y.\n\n" \
                "Summary of learning as a numbered list:"
    else:
        task_prompt = "Generate a numbered list of general advice based on your previous attempt, that will help the robot to successfully accomplish the SAME task in the SAME environment in future.\n"
    
    final_prompt = summary_prompt + '\n\n' + meta_summary_prompt + '\n\n' + task_prompt


    # "The CURRENT TRACE can be accompanied with PRECISION_FINAL, RECALL_FINAL, REWARD_FINAL, which are defined as below:\n" \
    # "PRECISION_FINAL: The fraction of final state of the world from the agent's attempt overlap with the desired final state of the world. This estimates how precise/accurate were the agent's actions to achieve the desired final state of the world for the task. It can be between 0 (lowest) to 100 (highest). A lower PRECISION_FINAL means actions taken by the agent were not useful for solving the task.\n" \
    # "RECALL_FINAL: The fraction of the desired final state of the world achieved by final state of world resulted from the agent's attempt. This estimates the coverage/comprehensiveness of the agent's actions to achieve the desired final state of the world for the task. It can be between 0 (lowest) to 100 (highest). A lower RECALL_FINAL means actions taken by the agent were not useful for solving the task.\n" \
    # "REWARD_FINAL: The final reward the agent received after its attempt to the task. It can be between -100 (lowest) to 100 (highest).\n\n" \


    # check trace length
    total_budget = 7500 - 1000
    tokens_insturction = getTokenLength(final_prompt)
    total_budget -= tokens_insturction
    trace = get_trace(current_run)
    tokens_trace = getTokenLength(trace)
    
    memories = {'prev_memory': prev_runs_list, 'meta_memory': meta_summary if meta_summary else "", 'trace': trace}

    prev_memories = ""
    if prev_runs_list:
        prev_memories = format_memory(memories['prev_memory'])
    if quadrant == 2:
        if meta_summary:
            prev_memories = "META LEARNINGS:\n" + memories['meta_memory'] + prev_memories

    tokens_prev_mem = getTokenLength(prev_memories)

    while tokens_trace + tokens_prev_mem > total_budget:
        if len(memories['meta_memory']) > 0:
            memories['meta_memory'] = ""
            tokens_prev_mem -= getTokenLength(meta_summary)
        elif len(memories['prev_memory']) > 0:
            tokens_prev_mem -= getTokenLength(format_memory([memories['prev_memory'][0]]))
            memories['prev_memory'] = memories['prev_memory'][1:]
        else:
            trace = get_trace(current_run, truncate=True)
            tokens_trace = getTokenLength(trace)
            if tokens_trace + tokens_prev_mem > total_budget:
                trace = ' '.join(trace.split(' ')[(total_budget - tokens_trace - tokens_prev_mem):])
            memories['trace'] = trace
                
    final_trace = memories['trace'] + "\n\nList of advice:" if quadrant == 0 else ""
    final_prev_memories = ""
    if memories['prev_memory']:
        final_prev_memories = format_memory(memories['prev_memory'])
    if quadrant == 2:
        if len(memories['meta_memory']) > 0:
            final_prev_memories = "META LEARNINGS:\n" + memories['meta_memory'] + final_prev_memories

    summary = summarize(final_trace, final_prompt, system_prompt, prev_memories=final_prev_memories,
                        model=model, temp=temp, tokens=1000)

    print("SUMMARY: {}".format(summary))
    return summary


def get_change_in_states(run_history):
    if not run_history:
        return None

    # print(data.keys())
    # print(f"Task: {data['taskDescription']}")
    # print(f"history[0]: {data['history'][0]}")

    prev_room = ''
    prev_freelook_room_states = []
    prev_freelook_object_states = []
    all_steps_state_changes = []
    for step_id, step in enumerate(run_history, start=0):
        observation = step['observation'].split('.')[0]
        freelook = step['freelook']

        freelook_part1 = freelook.split('\nYou also see:')[0]
        room_name = freelook_part1.split('. In it, you see:')[0].replace('This room is called the ', '')
        freelook_room_states_str = []
        if '. In it, you see:' in freelook_part1:
            for f in freelook_part1.split('. In it, you see:')[1].split('\n\t'):
                if f.strip():
                    freelook_room_states_str.append(f.strip())
        freelook_room_states = [
            f'In the {room_name} : {s}' for s in
            freelook_room_states_str
        ]

        freelook_part2 = freelook.split('\nYou also see:')[1]
        freelook_object_states = []
        for f in freelook.split('\nYou also see:')[1].split('\n\t'):
            if f.strip():
                freelook_object_states.append(f.strip())

        # append state changes to all_steps_state_changes
        if step_id:  # skip step-0 which is always 'look-around'
            all_steps_state_changes.append(observation)

            if prev_room == room_name:  # take a diff of states only when the room remains the same
                                        # when agent moves to another room, 'observation' captures the move
                                        # all object states in the new room need not be recorded as state changes.
                for f_r in freelook_room_states:
                    if f_r and f_r not in prev_freelook_room_states:
                        all_steps_state_changes.append(f_r)
            for f_o in freelook_object_states:
                if f_o and f_o not in prev_freelook_object_states:
                    all_steps_state_changes.append(f_o)

        # print(f"step-{step_id}: {step['action']}")
        # print(f"observation: {observation}")
        # print(f"freelook: {freelook}")
        # print(f"episode_state_changes: {all_steps_state_changes}")
        # print(f"\n==============\n")

        prev_room = room_name
        prev_freelook_room_states = freelook_room_states
        prev_freelook_object_states = freelook_object_states

    # print(f'\n--------------------\n')
    # print(f"final_room_states:{prev_freelook_room_states}")
    # print(f"final_object_states:{prev_freelook_object_states}")

    return {
        'all': all_steps_state_changes,
        'final': prev_freelook_room_states + prev_freelook_object_states
        }

def precision_recall_f1(gold, model):
    intersection = set(gold).intersection(set(model))

    if len(model):
        precision = len(intersection) * 1.0 / len(set(model))
        recall = len(intersection) * 1.0 / len(set(gold))
    else:
        precision, recall = 0.0, 0.0

    if precision == 0.0 and recall == 0.0:
        F1 = 0.0
    else:
        F1 = (2 * precision * recall) / (precision + recall)

    return {
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'F1': round(F1 * 100, 2)
    }