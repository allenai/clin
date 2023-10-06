import time
import random
import argparse
import os
import json

from scienceworld import ScienceWorldEnv

from model_utils import get_best_matched_action_using_sent_transformer,\
    get_clin_sw_next_action_multi_turn, \
    summarize_trace_for_preconditions_sTsW
import torch
from sentence_transformers import SentenceTransformer
import json


def clinAgent(args):
    """ CLIN agent """
    task_num = int(args['task_num'])
    var_num = int(args["var_num"])
    print(f"Running CLIN agent for taskIdx:{task_num}, varIdx:{var_num}")
    exitCommands = ["quit", "exit"]
    sent_transformer_model = SentenceTransformer('bert-base-nli-mean-tokens', device=args['device'])


    ## load gold summaries obtained from gold traces
    gold_path = args['gold_traces']
    gold_summaries = []
    if gold_path:
        if 'jsonl' in gold_path:
            with open(gold_path, 'r', encoding='utf-8') as f:
                for line in f:
                    gold_summaries.append(json.loads(line.rstrip('\n|\r')))
        elif 'json' in gold_path:
            gold_summaries.append(json.load(open(gold_path, 'r', encoding='utf-8')))
        else: # read all files in the gold directory
            print("Reading gold traces")
            for file_name in os.listdir(gold_path):
                if file_name.endswith('.json'):
                    f = open(os.path.join(gold_path, file_name))
                    try:
                        data = json.load(f)
                        gold_summaries.append(data)
                    except:
                        print("Could not read gold trace file")
                        break
        print("Done reading gold traces")
        
    gold_summaries_map = {}
    gold_histories_map = {}
    quadrant = args['quadrant']
    use_last_k_memories = args['use_last_k_memories']

    for sum in gold_summaries:
        # TODO: check partition
        if sum["taskIdx"] in gold_summaries_map:
            if quadrant == 1:
                gold_summaries_map[sum["taskIdx"]].update({sum["variationIdx"]: sum["summary"]})
            elif quadrant == 2:
                gold_summaries_map[sum["taskIdx"]].update({sum["variationIdx"]: sum["q2_summary"]})
            elif quadrant == 3:
                gold_summaries_map[sum["taskIdx"]].update({sum["variationIdx"]: sum["q3_summary"]})
            
            gold_histories_map[sum["taskIdx"]].update({sum["variationIdx"]: sum["history"]})
        else:
            if quadrant == 1:
                gold_summaries_map[sum["taskIdx"]] = {sum["variationIdx"]: sum["summary"]}
            elif quadrant == 2:
                gold_summaries_map[sum["taskIdx"]] = {sum["variationIdx"]: sum["q2_summary"]}
            elif quadrant == 3:
                gold_summaries_map[sum["taskIdx"]] = {sum["variationIdx"]: sum["q3_summary"]}
            
            gold_histories_map[sum["taskIdx"]] = {sum["variationIdx"]: sum["history"]}

    simplificationStr = args['simplification_str']
    numEpisodes = args['num_episodes']
    gpt_model = args['gpt_model']
    temperature = args['temperature']
    use_gold_memory_in_ep0 = bool(args['use_gold_memory_in_ep0'])
    summarize_end_of_episode = bool(args['summarize_end_of_episode'])

    # Keep track of the agent's final scores
    finalScores = []
    memory_of_runHistories = []
    output_dir = args['output_path_prefix']
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    summaryfname = args['output_path_prefix'] + "summary.txt"
    summaryFile = open(summaryfname, "w")
    env = ScienceWorldEnv("", args['jar_path'], envStepLimit=args['env_step_limit'])

    taskNames = env.getTaskNames()
    print("Task Names: " + str(taskNames))

    num_test_variations_to_run = 1
    for taskIdx in [task_num]:
        # Choose task
        taskName = taskNames[taskIdx]        # Just get first task
        env.load(taskName, 0, "")  
        print("Task Name: " + taskName)
        print("Task Description: " + str(env.getTaskDescription()))

        for varIdx in [var_num]:
            # Initialize environment
            prev_episode_summary_str = ""

            # Start running episodes
            for episodeIdx in range(0, numEpisodes):

                # Run histories
                runHistories = []

                # Save history -- and when we reach maxPerFile, export them to file
                filenameOutPrefix = f"{output_dir}/Task{taskIdx}_Var{varIdx}_Ep{episodeIdx}_runhistories.json"
                historyOutJSON = open(filenameOutPrefix, "w")

                env.load(taskName, varIdx, simplificationStr, generateGoldPath=True)

                # Reset the environment
                initialObs, initialDict = env.reset()

                # Example accessors
                templates, lut = env.getPossibleActionObjectCombinations()
                print("Task Name: " + taskName)
                print("Task Variation: " + str(varIdx))
                print("Task Description: " + str(env.getTaskDescription()) )

                gold_memory = ""
                if use_gold_memory_in_ep0:
                    if str(taskIdx) in gold_summaries_map:
                        if str(varIdx) in gold_summaries_map[str(taskIdx)]:
                            gold_memory = gold_summaries_map[str(taskIdx)][str(varIdx)]
                   
                score = 0.0
                score_positive = 0.0
                isCompleted = False
                curIter = 0

                # Run one episode until we reach a stopping condition (including exceeding the maximum steps)
                generated_action_str = "look around"        # First action
                generated_rationale_str = "look around"        # First action
                previous_rationales = []
                previous_actions = []
                previous_observations = []
                rationaleHistory = [""]
                subgoalHistory = [env.getGoalProgressJSON()]
                rawActionHistory = [generated_action_str]
                topNActionHistory = [{}]
                stepwise_prf_history = []
                sentenceTransformerRuntimes = []

                earlyStop = False
                while (generated_action_str not in exitCommands) and (isCompleted == False):
                    print("----------------------------------------------------------------")
                    print ("Step: " + str(curIter))

                    # Send user input, get response
                    observation, reward, isCompleted, info = env.step(generated_action_str)

                    previous_actions.append(generated_action_str)
                    previous_rationales.append(generated_rationale_str)
                    observation_str = observation.replace("\n", " ")                           # Keep the entire history
                    previous_observations.append(observation_str)

                    score = info['score']
                    if score > 0.0:
                        score_positive = score

                    # Store subgoal progress
                    subgoalHistory.append(env.getGoalProgressJSON())

                    print("\n>>> " + observation)
                    print("Reward: " + str(reward))
                    print("Score: " + str(score))
                    print("isCompleted: " + str(isCompleted))

                    # The environment will make isCompleted `True` when a stop condition has happened, or the maximum number of steps is reached.
                    if (isCompleted):
                        break

                    # get next_action predicted by CLIN multi-turn
                    summary_str = ""
                    if use_gold_memory_in_ep0 and episodeIdx == 0:
                        summary_str = gold_memory
                    elif summarize_end_of_episode:
                        summary_str = prev_episode_summary_str

                    generated_action_str = "N/A"
                    feedback = ""
                    num_retries = 0
                    max_num_retries_executor = 3
                    while (generated_action_str == "N/A" and num_retries < max_num_retries_executor):
                        num_retries += 1

                        response = get_clin_sw_next_action_multi_turn(
                                task=env.getTaskDescription(),
                                current_obs=env.look(),
                                current_inventory=env.inventory(),
                                objects_set=env.getPossibleObjects(),
                                next_actions_set=env.getPossibleActions(),
                                previous_rationales=previous_rationales,
                                previous_actions=previous_actions,
                                previous_observations=previous_observations,
                                model=gpt_model,
                                summary=summary_str,
                                temperature=temperature,
                                quadrant=quadrant,
                                feedback=feedback,
                                episodeIdx=episodeIdx
                            )

                        generated_rationale_str = response['response_str']
                        generated_action_str = response["pred_next_action"].lower().strip()
                        print("GPT4 generated action: " + str(generated_action_str))

                        if "TASK_COMPLETE" in response['response_str'] or \
                            "TASK COMPLETE" in response['response_str']or \
                            "TASKCOMPLETE" in response['response_str'] or \
                            "task complete" in  response['response_str'] or \
                            'successfully completed' in  response['response_str'] or \
                            'There is no further action required' in response['response_str']:
                            generated_action_str = "exit"
                            best_match_action = generated_action_str
                            topN = []
                        else:
                            valid_actions_list = info['valid']  # populated in the return from 'step'
                            valid_actions_list = [x for x in valid_actions_list if 'reset' not in x] # remove reset from valid actions

                            if "FOCUS" in response["pred_next_action"] or  "focus" in response["pred_next_action"]:
                                valid_actions_list = [x for x in valid_actions_list if 'focus' in x]
                            else:
                                valid_actions_list = [x for x in valid_actions_list if 'focus' not in x]
                            
                            best_match_score = 0.0
                            best_match_action = "exit"
                            if len(valid_actions_list) == 0:
                                # check "Ambiguous request in observation"
                                if "Ambiguous request" in observation:
                                    valid_actions_list = [str(x) for x in range(len(observation.split('\n')[1:]))]

                            if len(valid_actions_list) > 0:
                                # Time how long it takes to map generated next_action to one of the valid_actions?
                                start = time.time()
                                best_match_action, topN = get_best_matched_action_using_sent_transformer(
                                    allowed_actions=valid_actions_list,
                                    query=generated_action_str,
                                    model=sent_transformer_model,
                                    device=args['device']
                                )
                                end = time.time()
                                sentenceTransformerRuntimes.append(round(end - start, 2))

                                print("Sentence transformer runtimes: " + str(sentenceTransformerRuntimes))

                                # Check top-1 action match score, and if the score < threshold then 
                                best_match_score = topN[0][1]

                            if best_match_score > 0.9 or \
                                best_match_action in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'] or (num_retries==max_num_retries_executor) or (len(valid_actions_list) == 0) :
                                generated_action_str = best_match_action
                            else:
                                generated_action_str = "N/A"
                                feedback = "Your generated action '{}' cannot be matched to a valid action. Please retry with a different phrasing or a different action.".format(response['pred_next_action'])
                                print(f"ERROR: retry:{num_retries}\tgen={response['pred_next_action']}\tmatch={best_match_action}\tscore{topN} feedback={feedback}")
                    
                    # Reasoning rationale
                    print("Reasoning rationale: " + str(response["reasoningStr"]))
                    rationaleHistory.append(response["reasoningStr"])

                    
                    print("Next ChatGPT action: " + str(generated_action_str))
                    rawActionHistory.append(generated_action_str)

                    
                    topNActionHistory.append(topN)
                    print("Next ChatGPT action (mapped): " + str(generated_action_str))

                    # Keep track of the number of commands sent to the environment in this episode
                    curIter += 1

                    if (len(previous_actions) > 5):
                        # Check if the last 5 actions were all the same.  If so, exit
                        if (previous_actions[-1] == previous_actions[-2] == previous_actions[-3] == previous_actions[-4] == previous_actions[-5]):
                            print("Last 5 actions were the same.  Exiting.")
                            earlyStop = True
                            break

                        # Check if the max num steps (here, 1.5*gold_sequence_length) have been executed.  If so, exit
                        if len(previous_actions) >= 1.5*len(env.getGoldActionSequence()):
                            print("Model generated an action sequence which is 1.5 times longer than"
                                  "the gold action sequence without succeeding at the task.  Exiting.")
                            earlyStop = True
                            break

                print("Goal Progress:")
                print(env.getGoalProgressStr())
                time.sleep(1)

                # Episode finished -- Record the final scoref
                finalScores.append({
                    "taskIdx": taskIdx,
                    "taskName": taskName,
                    "variationIdx": varIdx,
                    "episodeIdx": episodeIdx,
                    "final_score": score,
                    "isCompleted": isCompleted

                })

                # Report progress of model
                print ("Final score: " + str(score))
                print ("isCompleted: " + str(isCompleted))

                # Show gold path
                gold_path = str(env.getGoldActionSequence())
                print("Gold Path:" + gold_path)

                # Get run history
                runHistory = env.getRunHistory()

                # Add rationales to run history
                for idx, rationaleStr in enumerate(rationaleHistory):
                    if idx >= len(runHistory['history']):
                        break
                    runHistory['history'][idx]['rationale'] = rationaleStr
                    runHistory['history'][idx]['rawAction'] = rawActionHistory[idx]
                    runHistory['history'][idx]['topNActions'] = topNActionHistory[idx]
                    runHistory['history'][idx]['subgoalProgress'] = subgoalHistory[idx]

                # Also store final score
                runHistory['episodeIdx'] = episodeIdx
                runHistory['finalScore'] = score
                runHistory['finalScorePositive'] = score_positive
                runHistory['isCompleted'] = isCompleted
                runHistory['earlyStop'] = earlyStop
                runHistory['model'] = gpt_model
                runHistory['gold-action-seq'] = gold_path
                runHistory['memory-seen'] = summary_str
                
                # Summarize learnings at the end of each episode and store them in the runHistory
                # use last k  memories to summarize learnings
                episode_summary_str = ""
                if summarize_end_of_episode:
                    prev_runs_list = memory_of_runHistories[-1 * use_last_k_memories:]
                    episode_summary_str = summarize_trace_for_preconditions_sTsW(runHistory,
                                                        prev_runs_list=prev_runs_list,
                                                        gold_run=None,
                                                        demo_examples=None,
                                                        model="gpt-4",
                                                        temp=temperature,
                                                        quadrant=quadrant,
                                                        meta_summary=gold_memory if quadrant == 2 else '',
                                                        use_last_k_memories=use_last_k_memories) # q2 summary passed as gold summary 
                runHistory['summary'] = episode_summary_str
                prev_episode_summary_str = episode_summary_str

                # Save runHistories into a JSON file
                print ("Writing history file: " + filenameOutPrefix)
                memory_of_runHistories.append(runHistory)
                json.dump(runHistory, historyOutJSON, indent=4, sort_keys=False)
                historyOutJSON.flush()
                historyOutJSON.close()

    # Show final episode scores to user:
    # avg = sum([x for x in finalScores if x >=0]) / len(finalScores)     # Clip negative scores to 0 for average calculation
    print("")
    print("---------------------------------------------------------------------")
    print(" Summary (ChatGPT Agent)")
    print(" Simplifications: " + str(simplificationStr))
    print("---------------------------------------------------------------------")
    print(f"task\tName\tvar\tepi\tscore\tcomplete?")
    for finalScore in finalScores:
        print(f"{finalScore['taskIdx']}\t{finalScore['taskName']}\t{finalScore['variationIdx']}\t"
              f"{finalScore['episodeIdx']}\t{finalScore['final_score']}\t{finalScore['isCompleted']}")
    print("---------------------------------------------------------------------")
    print("")

    print("Completed.")

    summaryFile.write(f"" + '\n')
    summaryFile.write("---------------------------------------------------------------------" + '\n')
    summaryFile.write(" Summary (ChatGPT Agent)" + '\n')
    summaryFile.write(" Simplifications: " + str(simplificationStr) + '\n')
    summaryFile.write("---------------------------------------------------------------------" + '\n')
    summaryFile.write(f"task\tName\tvar\tepi\tscore\tcomplete?" + '\n')
    for finalScore in finalScores:
        summaryFile.write(f"{finalScore['taskIdx']}\t{finalScore['taskName']}\t{finalScore['variationIdx']}\t"
              f"{finalScore['episodeIdx']}\t{finalScore['final_score']}\t{finalScore['isCompleted']}" + '\n')
    summaryFile.write("---------------------------------------------------------------------" + '\n')
    summaryFile.write("Completed." + '\n')
    summaryFile.close()


def build_simplification_str(args):
    """ Build simplification_str from args. """
    simplifications = list()
    if args["teleport"]:
        simplifications.append("teleportAction")

    if args["self_watering_plants"]:
        simplifications.append("selfWateringFlowerPots")

    if args["open_containers"]:
        simplifications.append("openContainers")

    if args["open_doors"]:
        simplifications.append("openDoors")

    if args["no_electrical"]:
        simplifications.append("noElectricalAction")

    return args["simplifications_preset"] or ",".join(simplifications)


#
#   Parse command line arguments
#
def parse_args():
    desc = "Run a model that chooses random actions until successfully reaching the goal."
    parser = argparse.ArgumentParser(desc)
    parser.add_argument("--jar_path", type=str,
                        help="Path to the ScienceWorld jar file. Default: use builtin.")
    parser.add_argument("--task-num", type=str, default="4",
                        help="Specify the task number to play. Default: %(default)s")
    parser.add_argument("--var-num", type=int, default=1,
                        help="Specify the task variation number to play. Default: %(default)s")
    parser.add_argument("--env-step-limit", type=int, default=100,
                        help="Maximum number of steps per episode. Default: %(default)s")
    parser.add_argument("--num-episodes", type=int, default=2,
                        help="Number of episodes to play. Default: %(default)s")
    parser.add_argument("--gpt-model", type=str, default="gpt-4-0613",
                        help="Choose GPT model to use ['gpt-3.5-turbo', 'gpt-4-0613']. Default: %(default)s")
    parser.add_argument("--summarize_end_of_episode", type=int, default=1,
                        help="Summarize at the end of episode (for preconditions)")
    parser.add_argument("--device", type=str, required=True,
                        help="Select device to be used by sentence transformer. ['cpu', 'cuda', 'cuda:0']")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Select temperature for running chatgpt completion api")
    parser.add_argument("--use-gold-memory-in-ep0", type=int, default=0,
                        help="Use gold memory and seed learnings in episode 0.")
    parser.add_argument("--gold-traces", type=str, default="",
                        help="Gold action sequences and corresponding observations.")
    parser.add_argument("--use-last-k-memories", type=int, default=3,
                        help="Use last k memories when summarizing learnings.")
    parser.add_argument("--quadrant", type=int,
                        help="Specify the quadrant in which the model is being evaluated."\
                        "1: (adapt) same task/same world, "\
                        "2: (gen-env) same task/different world, "\
                        "3: (gen-task) different task, same world")
    parser.add_argument("--output-path-prefix", default="save-histories",
                        help="Path prefix to use for saving episode transcripts. Default: %(default)s")
    
    simplification_group = parser.add_argument_group('Game simplifications')
    simplification_group.add_argument("--simplifications-preset", choices=['easy'],
                                      help="Choose a preset among: 'easy' (apply all possible simplifications).")
    simplification_group.add_argument("--teleport", action="store_true",
                                      help="Lets agents instantly move to any location.")
    simplification_group.add_argument("--self-watering-plants", action="store_true",
                                      help="Plants do not have to be frequently watered.")
    simplification_group.add_argument("--open-containers", action="store_true",
                                      help="All containers are opened by default.")
    simplification_group.add_argument("--open-doors", action="store_true",
                                      help="All doors are opened by default.")
    simplification_group.add_argument("--no-electrical", action="store_true",
                                      help="Remove the electrical actions (reduces the size of the action space).")

    args = parser.parse_args()
    params = vars(args)
    return params


def main():
    print("ScienceWorld 1.0 API Examples - CLIN Agent")
    # Parse command line arguments
    args = parse_args()
    args["simplification_str"] = build_simplification_str(args)
    clinAgent(args)


if __name__ == "__main__":
    main()
