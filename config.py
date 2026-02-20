import os
from typing import Dict

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


llama_config = {
    "model_name": "meta-llama/Llama-2-13b-chat-hf",
    "model_id": "meta-llama/Llama-2-13b-chat-hf",
    "cache_dir": None,  # f"{ROOT_PATH}/models_cache/",
    "max_new_tokens": 1024,
    "repetition_penalty": 1.1,
    "temperature": 0.7,
    "do_sample": False,
    "max_input_length": 1024,
    "SEED": 42,
}

zephyr_config = {
    "model_name": "stabilityai/stablelm-zephyr-3b",  # "stabilityai/stablelm-zephyr-3b",  # HuggingFaceH4/zephyr-7b-beta
    "model_id": "stabilityai/stablelm-zephyr-3b",  # "stabilityai/stablelm-zephyr-3b",  # HuggingFaceH4/zephyr-7b-beta
    "cache_dir": None,  # f"{ROOT_PATH}/models_cache/",
    "max_new_tokens": 4096,
    "repetition_penalty": 1.1,
    "temperature": 0.7,
    "do_sample": True,
    "max_input_length": 2048,
    "SEED": 42,
}

evaluation_config = {
    "cache_dir": f"{ROOT_PATH}/models_cache/",
}


retrieval_config = {
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "experiment_name": "retrieval",
    "experiment_path": f"{ROOT_PATH}/results/",
    "results_file": "retrieval_user_query.csv",
    "query_gen_results_file": "generated_queries_4shot_4q_retrieved_docs_rerank.csv",
    "generated_queries_file": f"{ROOT_PATH}/results/RTG_generated_queries/generated_queries_4shot_4q.csv",
    "posthoc_retrieval_file": f"{ROOT_PATH}/results/llama13/G/answer_generation_G.csv",
    "results_file_posthoc": f"{ROOT_PATH}/results/llama13/G/answer_generation_GTR.csv",
    "query_aggregation": "rerank",  # can be : "rerank",  "seperate_queries", vote, sort, simple, summed_vote, mean_vote, combSum
    "filter_queries": False,
    "nb_passages": 5,
}

prompts_config = {
    "prompt_with_gold": {
        "system": "You are an assistant that provides answers and the source of the answer. I will give a question and several context texts about the question. Choose the context texts that are most relevant to the questions and based on them, give a short answer to the question. You must provide in-line citations to each statement in the answer from the context. The citations should appear as numbers within brackets [] such as [1], [2] based on the given contexts. A statement may need to be supported by multiple contexts and should then be cited as [1] [2].",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt": {
        "system": "You are an assistant that provides answers and the source of the answer. I will give a question and several context texts about the question. Choose the context texts that are most relevant to the questions and based on them, give a short answer to the question. You must provide in-line citations to each statement in the answer from the context. The citations should appear as numbers within brackets [] such as [1], [2] based on the given contexts. A statement may need to be supported by multiple contexts and should then be cited as [1] [2].",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompot_without_context": {
        "system": "You are an assistant that provides answers. I will give you a question and based on your knowledge, give a brief answer to the question.",
        "user": "QUESTION: {query} \n\n ANSWER:",
    },
    "prompt_without_citation": {
        "system": "You are an assistant that provides answers. I will give a question and several context texts about the question. Choose the context texts that are most relevant to the questions and based on them, give a short answer to the question.",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "query_gen_prompt": {
        "system": "You are an assistant that helps the user with their search. I will give you a question, and based on this question, you will suggest other specific queries that help retrieve documents that contain the answer. Only generate your suggested queries without explanation. The maximum number of queries is 4.",  # "You are an assistant that helps the user with their search. I will give you a question and its answer, based on this question and the answer, you will suggest other specific queries that help retrieve documents that contain the answer. The maximum number of queries is 4.",
        "user_with_answer": "QUESTION: {query} \n\n ANSWER:\n {answer} \n\n SUGGESTED QUERIES:",
        "user": "QUESTION: {query} \n\n SUGGESTED QUERIES:",
    },
}

exp_query_gen_fewshots_hagrid = {
    "experiment_name": "RTG_generated_queries",
    "experiment_path": f"{ROOT_PATH}/results/",
    "results_file": "generated_queries_4shot_4q.csv",  # "generated_queries_4shot_4q.csv",
    "config_file": "generated_queries_4shot_4q_config.json",  # "generated_queries_4shot_4q_config.json",
    "setting": "fewshot",  # zeroshot
    "query_gen_prompt": {
        "system": "You are an assistant that helps the user with their search. I will give you a question, based on the possible answer of the question you will provide queries that will help find documents that support it. Only generate your suggested queries without explanation. The maximum number of queries is {nb_queries}",  # .
        "user_with_answer": "QUESTION: {query} \n\n ANSWER:\n {answer} \n\n SUGGESTED QUERIES:",
        "user": "QUESTION: {query} \n\n SUGGESTED QUERIES:",
    },
    "include_answer": False,
    "nb_queries_to_generate": 4,
    "nb_shots": 4,
    "fewshot_examples": [
        {
            "user_query": "Why does milk need to be pasteurized?",
            "generated_queries": [
                "How does pasteurization work to make milk safer?",
                "What are the arguments that support milk pasteurization?"
                "What is the purpose of milk pasteurization?",
                "Adoption of milk pasteurization in developed countries",
                "What are the differences between pasteurization and sterilization in milk processing?",
                "United States raw milk debate",
            ],
        },
        {
            "user_query": "What is the largest species of rodent?",
            "generated_queries": [
                "Is there a rodent bigger than a capybara?",
                "How big are giant hutia vs capybara?",
                "Comparison of capybara with other rodent species",
                "the world's heaviest rodent species",
                "Sizes of common rodents",
            ],
        },
        {
            "user_query": "What was the first animal to be domesticated by humans?",
            "generated_queries": [
                "Examples of early domesticated animals",
                "Commensal pathway of dog domestication",
                "First animal domesticated by humans",
                "When did the domestication process of animals first start?",
                "Is there evidence supporting the domestication of dogs?",
                "What were the first steps of animal domestication?",
                "What started the dog domestication process?",
            ],
        },
        {
            "user_query": "When is All Saints Day?",
            "generated_queries": [
                "What do Catholics do on All Saints Day?",
                "Observance of All Saints Day by non-Catholic Christians",
                "Solemnity of All Saints Day and its transfer to Sundays",
                "What is the difference between Day of the Dead and All Saints Day?",
                "Is All Saints Day a Catholic holy day?",
                "Do Protestants celebrate All Saints Day?",
            ],
        },
        {
            "user_query": "Who won the battle of Trafalgar?",
            "generated_queries": [
                "Napoleonic Wars duration after the Battle of Trafalgar",
                "How did the victory at the Battle of Trafalgar impact the Napoleonic Wars?",
                "Was Napoleon defeated at Trafalgar?",
                "What were the strategic implications of the British victory at the Battle of Trafalgar?",
                "Impact of the Battle of Trafalgar on British naval supremacy",
                "Paintings commemorating the Battle of Trafalgar",
            ],
        },
        {
            "user_query": "What was the last Confederate victory in the Civil War?",
            "generated_queries": [
                "What was the final major battle for the Confederacy?",
                "When was the Battle of Natural Bridge fought?",
                "Battle of Plymouth as an earlier Confederate victory",  ### was not used in first tests
                "Battle of Palmito Ranch",
                "Was Battle of Palmito Ranch the final battle of the American Civil War?",
                "Comparison of Battle of Palmito Ranch with other Confederate victories",
            ],
        },
    ],
}

exp_query_gen_fewshots_from_alce = {
    "experiment_name": "zephyr_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/",
    "results_file": "generated_queries_4shot_4q_asqa_llama_asqashots.csv",  # "generated_queries_4shot_4q.csv",
    "config_file": "generated_queries_4shot_4q_asqa_config_llama_asqashots.json",  # "generated_queries_4shot_4q_config.json",
    "setting": "fewshot",  # zeroshot
    "query_gen_prompt": {
        "system": "You are an assistant that helps the user with their search. I will give you a question, based on the possible answer of the question you will provide queries that will help find documents that support it. Only generate your suggested queries without explanation. The maximum number of queries is {nb_queries}",  # .
        "user_with_answer": "QUESTION: {query} \n\n ANSWER:\n {answer} \n\n SUGGESTED QUERIES:",
        "user": "QUESTION: {query} \n\n SUGGESTED QUERIES:",
    },
    "include_answer": False,
    "nb_queries_to_generate": 4,
    "nb_shots": 4,
    "fewshot_examples": [
        {
            "user_query": "Who published harry potter and the prisoner of azkaban?",
            "generated_queries": [
                "Who published harry potter and the prisoner of azkaban in the UK?",
                "Who published harry potter and the prisoner of azkaban in the US?",
                "Who published harry potter and the prisoner of azkaban in Canada?",
            ],
        },
        {
            "user_query": "Where does the vikings play their home games?",
            "generated_queries": [
                "What stadium does the vikings play their home games since 2016?",
                "Where is the stadium that the vikings play their home games since 2013?",
                "What stadium does the vikings play their home games from 1982-2013?",
                "What stadium does the vikings play their home games in 2014 and 2015?",
                "What stadium does the vikings play their home games in from 1961-1981?",
            ],
        },
        {
            "user_query": "When was the last time a us submarine sunk?",
            "generated_queries": [
                "When was the last time a us nuclear submarine sunk?",
                "When was the last time a decommissioned us submarine sunk?",
                "When was the last time a us submarine sunk prior to commissioning?",
                "When was the last time a us non-nuclear submarine sunk?",
            ],
        },
    ],
}

llms_config = {"zephyr": zephyr_config, "llama": llama_config}
architectures_config = {
    "G": {
        "use_retrieved": False,
        "hagrid_gold": False,
        "retrieved_passages_file": None,
        "use_context": False,
        "nb_passages": 0,
        "citation": False,
        "experiment_name": "G",
        "experiment_path": f"{ROOT_PATH}/results/llama13/",
        "results_file": "answer_generation_G.csv",
        "config_file": "config_answer_generation_G.json",
    },
    "RTG-gold": {
        "use_retrieved": False,
        "hagrid_gold": True,
        "retrieved_passages_file": None,
        "use_context": True,
        "nb_passages": None,
        "citation": True,
        "experiment_name": "RTG_gold",
        "experiment_path": f"{ROOT_PATH}/results/llama13/",
        "results_file": "answer_generation_RTG_gold_passages.csv",
        "config_file": "config_answer_generation_RTG_gold_passages.json",
    },
    "RTG-vanilla": {
        "use_retrieved": True,
        "hagrid_gold": False,
        "retrieved_passages_file": f"{retrieval_config['experiment_path']}{retrieval_config['experiment_name']}/{retrieval_config['results_file']}",  # f"{ROOT_PATH}/results/retrieval/retrieval_user_query.csv",
        "use_context": True,
        "nb_passages": 2,
        "citation": True,
        "experiment_name": "RTG_vanilla",
        "experiment_path": f"{ROOT_PATH}/results/llama13/",
        "results_file": "generation_RTG_vanilla_2_passages.csv",
        "config_file": "config_generation_RTG_vanilla_2_passages.json",
    },
    "RTG-query-gen": {
        "use_retrieved": True,
        "hagrid_gold": False,
        "retrieved_passages_file": f"{retrieval_config['experiment_path']}{retrieval_config['experiment_name']}/{retrieval_config['query_gen_results_file']}",  # f"{ROOT_PATH}/results/retrieval/generated_queries_4shot_4q_Hagrid_llama_retrieved_docs_rerank.csv",
        "use_context": True,
        "nb_passages": 2,
        "citation": True,
        "experiment_name": "RTG_generated_queries",
        "experiment_path": f"{ROOT_PATH}/results/llama13/",
        "results_file": "answer_generation_RTG_gen_queries_4q_4shots_rerank_2_passages.csv",
        "config_file": "config_answer_generation_RTG_gen_queries_4q_4shots_rerank_2_passages.json",
    },
    "GTR": {
        "retrieved_passages_file": f"{ROOT_PATH}/results/retrieval/generated_queries_4shot_4q_Hagrid_llama_retrieved_docs_rerank.csv",  # generated_queries_4shot_4q_rerank.csv",  # devMiracl_results_MonoT5_BM500_20_normal_corpus.csv",
        "nb_passages": 1,
        "experiment_name": "GTR",
        "experiment_path": f"{ROOT_PATH}/results/llama13/",
        "results_file": "answer_generation_GTR_1doc_per_sent.csv",
        "config_file": "answer_generation_GTR_1doc_per_sent.json",
        "posthoc_retrieval_file": f"{ROOT_PATH}/results/G/llama3/answer_generation_G.csv",
        "results_file_posthoc": f"{ROOT_PATH}/results/G/llama3/answer_generation__GTR.csv",
    },
}

CONFIG: Dict = {
    "architectures": architectures_config,
    "langauge_model": llms_config,  # add/modify LLM parameters : temeprature, max tokens, etc
    "dataset": "HAGRID",  # values: HAGRID or other : ALCE,..
    "data_path": None,  # For ALCE must provide path, values: None, f"{ROOT_PATH}/alce_data/asqa_eval_gtr_top100.json", ..
    "prompts": prompts_config,
    "retrieval": retrieval_config,
    "query_generation": exp_query_gen_fewshots_hagrid,
    "evaluation": evaluation_config,
    "multiple_gold_answers": False,
    "column_names": {
        "prediction": "output",  # values: output, generated_text,..
        "reference": "answers",  # values: answers (HAGRID), annotations (ALCE),  gold_truth
        "multiple_answers": "answer", #"answer",  # If multiple answers are possible, how to access the answers. For example if dataset has column 'gold_answers' which is a list of dictionarieies [{"answer":...}] then provide 'answer' here. Could be : answer (HAGRID), "long_answer" (ALCE)
        "passages": "quotes",  # values: retrieved_quotes (HAGRID), docs (ALCE)
        "gold_passages": "gold_quotes",  # values: None, quotes(HAGRID),  docs (ALCE), gold_quotes,
        "query": "query",  # values: query , question
    },
}
