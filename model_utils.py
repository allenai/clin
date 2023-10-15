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
        
    indices_actions_sorted_desc_word_sim = np.argsort(word_sim)
    if "cuda" in device:
        max_filtered_actions = 100000
    else:
        max_filtered_actions = 1000

    allowed_actions_filtered = [allowed_actions[ind] for ind in indices_actions_sorted_desc_word_sim[:max_filtered_actions]]
    print(f"actions_sorted_desc_word_sim: {allowed_actions_filtered[0:10]}")
    print(f"size of allowed_actions_filtered: {len(allowed_actions_filtered)}")

    # Second pass: Use the sentence transformer to find the best-matched action
    action_list_embeddings = model.encode(allowed_actions_filtered)
    query_embeddings = model.encode([query])
    sim = cosine_similarity(
        query_embeddings,
        action_list_embeddings
    )
    max_id = np.argmax(sim)

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

    print(f"size of allowed_actions_filtered: {len(allowed_actions_filtered)}")

    return allowed_actions_filtered[max_id], top5_action_sim_tuples


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
                                       summary="",
                                       temperature=0.0,
                                       quadrant=1,
                                       feedback="",
                                       episodeIdx=None):
    # We first ask the model to geberate goal (rationale) and then generate next action

    next_action_query = ""

    # Always have task information as first message
    sw_prompt_task = f"I'd like you to work your way through a virtual world to complete a particular task. " \
                     f"At each step, tell me which action you want to do, e.g., pick up something, " \
                     f"open something, look around etc. and I will tell you the result. " \
                     f"Then tell me the next action you want to do, until you complete the task." \
                     f"\n\n" \
                     f"Task: {task}" \
                     f"\n\n"
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

    next_action_query += f"Here is what you currently see:" \
                 f"\n" \
                 f" {current_obs}" \
                 f"Here is what is currently in your inventory:" \
                 f"\n" \
                 f" {current_inventory}" \
                 f"\n\n"

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

    return response

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
            final_score = m['finalScore']
            memory_string += "REWARD_FINAL: {}. This means: {}\n".format(final_score, success_map('reward', final_score))
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

    final_trace = memories['trace'] 
    final_trace += "\n\nList of advice:" if quadrant == 0 else ""
    final_prev_memories = ""
    if memories['prev_memory']:
        final_prev_memories = format_memory(memories['prev_memory'])
    if quadrant == 2:
        if len(memories['meta_memory']) > 0:
            final_prev_memories = "META LEARNINGS:\n" + memories['meta_memory'] + final_prev_memories

    summary = summarize(trace=final_trace, summary_prompt=final_prompt, 
                        system_prompt=system_prompt, prev_memories=final_prev_memories,
                        model=model, temp=temp, tokens=1000)

    print("SUMMARY: {}".format(summary))
    return summary
