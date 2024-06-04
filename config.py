import os
from typing import Dict

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


llama_config = {
    "model_name": "Llama-2-7b-chat-hf",
    "model_id": "meta-llama/Llama-2-7b-chat-hf",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "local_path_to_model": f"{ROOT_PATH}/models_cache/Llama-2-7b-chat-hf/",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "max_new_tokens": 4096,
    "repetition_penalty": 1.1,
    "do_sample": False,
    "max_input_length": 2048,
}
experiment = {
    "experiment_name": "llama_2_70b_query_with_full_instruction_no_passage_1st_sent",
    "experiment_path": f"{ROOT_PATH}/results/llms/",
    "results_file": "llama_2_70b_full_instruction_no_passage_1st_sent.csv",
    "config_file": "llama_2_70b_full_instruction_no_passage_1st_sentt_config.json",
    "support_passages": False,
    "prompt": "Tell me everything you know about {}. \n ANSWER:",  # "{}.",  # "{}. \n ANSWER:", #"Tell me everything you know about {}. \n ANSWER:",
}


hagrid_miracl_config = {
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "results_folder": f"{ROOT_PATH}/results/retrieval/miracl_dataset",
    "results_file": f"{ROOT_PATH}/results/retrieval/miracl_dataset/llama2_RTG_10_passages.csv",  # f"{ROOT_PATH}/results/retrieval/miracl_dataset/generated_queries_4shot_4q_lbre_nb_example_pmpt3_desc_rerank.csv", #f"{ROOT_PATH}/results/retrieval/miracl_dataset/gtr_posthoc_from_g_sent_sent.csv",#f"{ROOT_PATH}/results/retrieval/miracl_dataset/generated_queries_4shot_4q_lbre_nb_example_pmpt2_desc_rerank.csv",  # f"{ROOT_PATH}/results/retrieval/miracl_dataset/gtr_posthoc_from_g_fullAnswer.csv",
    "topics_file": f"{ROOT_PATH}/data/miracl_qrels/miracl-v1.0-en_topics_topics.miracl-v1.0-en-dev.tsv",
    "qrels_file": f"{ROOT_PATH}/data/miracl_qrels/miracl-v1.0-en_qrels_qrels.miracl-v1.0-en-dev.tsv",
    "corpus_path": f"{ROOT_PATH}/results/retrieval/miracl_dataset/corpus_dictionary.pkl",
    "generated_queries_file": f"{ROOT_PATH}/results/llms/llama2_zs_query_generation/llama2_manual_select_4shot_4qinE_10qtoGen_prmpt2.csv",  # f"{ROOT_PATH}/results/llms/zephyr_zs_query_generation/zephyr_relevant_4shot_4q_to_generate_prmpt3_4qinE.csv",  # f"{ROOT_PATH}/results/llms/zephyr_zs_query_generation/zephyr_relevant_4shot_4q_to_generate_prmpt2_VaryingQinEx.csv",  # zephyr_zs_query_generation_prompt2.csv",  # zephyr_zs_query_generation.csv",zephyr_fewshot_train_query_generation
    "posthoc_retrieval_file": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/llama2_zs_hagrid_answer_gen/Generate_vanilla/llama2_zs_answer_generation_generate_vanilla_zs.csv",
    "query_aggregation": "rerank",  # "rerank",  # "seperate_queries",  # "rerank",  # vote, sort, rerank, simple, summed_vote, mean_vote combSum
    "filter_queries": False,
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

exp_zephyr_answer_RTG_gen_queries = {
    "model_id": "stabilityai/stablelm-zephyr-3b",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "SEED": 42,
    "dataset": "HAGRID",
    "use_retrieved": True,
    "retrieved_passages_file": f"{ROOT_PATH}/results/retrieval/miracl_dataset/generated_queries_4shot_4q_lbre_nb_example_pmpt3_desc_rerank.csv",  # devMiracl_results_MonoT5_BM500_20_normal_corpus.csv",  # generated_queries_4shot_4q_lbre_nb_example_pmpt2_desc_seperate
    "use_context": True,
    "nb_passages": 10,
    "citation": True,
    "experiment_name": "RTG_generated_queries",  # "zephyr_zs_hagrid_ctxt_citing",  # "zephyr_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/zephyr_zs_hagrid_answer_gen/",
    "results_file": "zephyr_zs_answer_generation_RTG_gen_queries_4q_4shots_pmpt3_rerank_10_passages.csv",
    "config_file": "zephyr_zs_answer_generation_RTG_gen_queries_4q_4shots_pmpt3_rerank_10_passages.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}

exp_zephyr_answer_RTG_user_query = {
    "model_id": "stabilityai/stablelm-zephyr-3b",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "SEED": 42,
    "dataset": "HAGRID",
    "use_retrieved": True,
    "retrieved_passages_file": f"{ROOT_PATH}/results/retrieval/miracl_dataset/devMiracl_results_MonoT5_BM500_20_normal_corpus.csv",  # devMiracl_results_MonoT5_BM500_20_normal_corpus.csv",  # generated_queries_4shot_4q_lbre_nb_example_pmpt2_desc_rerank
    "use_context": True,
    "nb_passages": 11,
    "citation": True,
    "experiment_name": "RTG_user_query",  # "zephyr_zs_hagrid_ctxt_citing",  # "zephyr_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/zephyr_zs_hagrid_answer_gen/",
    "results_file": "zephyr_zs_answer_generation_RTG_user_query_10_passages.csv",
    "config_file": "zephyr_zs_answer_generation_RTG_user_query_10_passages.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}

exp_zephyr_answer_with_gold_passages = {
    "model_id": "stabilityai/stablelm-zephyr-3b",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "SEED": 42,
    "dataset": "HAGRID",
    "use_retrieved": False,
    "retrieved_passages_file": None,
    "use_context": True,
    "nb_passages": None,
    "citation": True,
    "experiment_name": "RTG_gold",  # "zephyr_zs_hagrid_ctxt_citing",  # "zephyr_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/zephyr_zs_hagrid_answer_gen/",
    "results_file": "zephyr_zs_answer_generation_RTG_gold_passages.csv",
    "config_file": "zephyr_zs_answer_generation_RTG_gold_passages.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}

exp_zephyr_answer_without_context = {
    "model_id": "stabilityai/stablelm-zephyr-3b",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "SEED": 42,
    "dataset": "HAGRID",
    "use_retrieved": False,
    "retrieved_passages_file": None,
    "use_context": False,
    "nb_passages": None,
    "citation": False,
    "experiment_name": "Generate_vanilla",  # "zephyr_zs_hagrid_ctxt_citing",  # "zephyr_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/zephyr_zs_hagrid_answer_gen/",
    "results_file": "zephyr_zs_answer_generation_generate_vanilla_zs.csv",
    "config_file": "zephyr_zs_answer_generation_generate_vanilla_zs.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}

exp_zephyr_answer_without_citation_gold = {
    "model_id": "stabilityai/stablelm-zephyr-3b",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "SEED": 42,
    "dataset": "HAGRID",
    "use_retrieved": False,
    "retrieved_passages_file": None,  # f"{ROOT_PATH}/results/retrieval/miracl_dataset/devMiracl_results_MonoT5_BM500_20_normal_corpus.csv",
    "use_context": True,
    "nb_passages": 10,
    "citation": False,
    "experiment_name": "zephyr_zs_answer_generation",  # "zephyr_zs_hagrid_ctxt_citing",  # "zephyr_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/",
    "results_file": "zephyr_zs_answer_generation_no_citation.csv",
    "config_file": "config_zephyr_zs_answer_generation_no_citation.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}

exp_zephyr_answer_without_citation = {
    "model_id": "stabilityai/stablelm-zephyr-3b",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "SEED": 42,
    "dataset": "HAGRID",
    "use_retrieved": True,
    "retrieved_passages_file": f"{ROOT_PATH}/results/retrieval/miracl_dataset/devMiracl_results_MonoT5_BM500_20_normal_corpus.csv",  # devMiracl_results_MonoT5_BM500_20_normal_corpus.csv",  # generated_queries_4shot_4q_lbre_nb_example_pmpt2_desc_rerank
    "use_context": True,
    "nb_passages": 10,
    "citation": True,
    "experiment_name": "RTG_user_query",  # "zephyr_zs_hagrid_ctxt_citing",  # "zephyr_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/zephyr_zs_hagrid_answer_gen/",
    "results_file": "zephyr_zs_answer_generation_RTG_user_query_no_cite_10_passages.csv",
    "config_file": "zephyr_zs_answer_generation_RTG_user_query_no_cite_10_passages.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}


exp_zephyr_query_gen_fewshots_train_prmt3 = {
    "model_id": "stabilityai/stablelm-zephyr-3b",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "SEED": 42,
    "dataset": "HAGRID",
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
    "test_examples": False,
    "experiment_name": "zephyr_zs_query_generation",  # "zephyr_zs_hagrid_ctxt_citing",  # "zephyr_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/",
    "results_file": "zephyr_relevant_4shot_4q_to_generate_prmpt3_4qinE.csv",
    "config_file": "zephyr_relevant_4shot_4q_to_generate_prmpt3_4qinE.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": None,
    },
    "description": "Few shot learning examples from Hagrid's train set using a prompt that indicates to use the answer. The example queries are chosen according to their relevance. Nb of queries in the examples is varied but instuction specifies nb of queries to gen",
}


exp_zephyr_answer_GTR = {
    "model_id": "stabilityai/stablelm-zephyr-3b",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "SEED": 42,
    "dataset": "HAGRID",
    "use_retrieved": False,
    "retrieved_passages_file": f"{ROOT_PATH}/results/retrieval/miracl_dataset/zephyr_gtr_posthoc_from_g_sent_sent.csv",
    "use_context": False,
    "nb_passages": None,
    "citation": False,
    "experiment_name": "GTR",  # "zephyr_zs_hagrid_ctxt_citing",  # "zephyr_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/zephyr_zs_hagrid_answer_gen/",
    "results_file": "zephyr_zs_hagrid_answer_GTR_10_passage_sent_sent.csv",
    "config_file": "zephyr_zs_hagrid_answer_GTR_10_passage_sent_sent.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}

exp_llama2_answer_RTG_user_query = {
    "model_name": "Llama-2-7b-chat-hf",
    "model_id": "meta-llama/Llama-2-7b-chat-hf",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "SEED": 42,
    "dataset": "HAGRID",
    "use_retrieved": True,
    "retrieved_passages_file": f"{ROOT_PATH}/results/retrieval/miracl_dataset/devMiracl_results_MonoT5_BM500_20_normal_corpus.csv",  # devMiracl_results_MonoT5_BM500_20_normal_corpus.csv",  # generated_queries_4shot_4q_lbre_nb_example_pmpt2_desc_rerank
    "use_context": True,
    "dataset": "HAGRID",
    "nb_passages": 10,
    "citation": True,
    "experiment_name": "RTG_user_query",  # "zephyr_zs_hagrid_ctxt_citing",  # "zephyr_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/llama2_zs_hagrid_answer_gen/",
    "results_file": "llama2_zs_answer_generation_RTG_user_query_10_passages.csv",
    "config_file": "llama2_zs_answer_generation_RTG_user_query_10_passages.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}

exp_llama2_answer_without_context = {
    "model_name": "Llama-2-7b-chat-hf",
    "model_id": "meta-llama/Llama-2-7b-chat-hf",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "use_retrieved": False,
    "SEED": 42,
    "dataset": "HAGRID",
    "retrieved_passages_file": None,
    "use_context": False,
    "nb_passages": None,
    "citation": False,
    "experiment_name": "Generate_vanilla",  # "zephyr_zs_hagrid_ctxt_citing",  # "zephyr_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/llama2_zs_hagrid_answer_gen/",
    "results_file": "llama2_zs_answer_generation_generate_vanilla_zs.csv",
    "config_file": "llama2_zs_answer_generation_generate_vanilla_zs.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}

exp_llama2_answer_with_gold_passages = {
    "model_name": "Llama-2-7b-chat-hf",
    "model_id": "meta-llama/Llama-2-7b-chat-hf",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "use_retrieved": False,
    "SEED": 42,
    "dataset": "HAGRID",
    "retrieved_passages_file": None,
    "use_context": True,
    "nb_passages": None,
    "citation": True,
    "experiment_name": "RTG_gold",
    "experiment_path": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/llama2_zs_hagrid_answer_gen/",
    "results_file": "llama2_zs_answer_generation_RTG_gold_passages.csv",
    "config_file": "llama2_zs_answer_generation_RTG_gold_passages.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}

exp_llama2_answer_GTR = {
    "model_name": "Llama-2-7b-chat-hf",
    "model_id": "meta-llama/Llama-2-7b-chat-hf",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "use_retrieved": False,
    "retrieved_passages_file": f"{ROOT_PATH}/results/retrieval/miracl_dataset/llama2_gtr_posthoc_from_g_sent_sent.csv",
    "use_context": False,
    "SEED": 42,
    "dataset": "HAGRID",
    "nb_passages": None,
    "citation": False,
    "experiment_name": "GTR",  # "llama2_zs_hagrid_ctxt_citing",  # "llama2_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/llama2_zs_hagrid_answer_gen/",
    "results_file": "llama2_zs_hagrid_answer_GTR_10_passage_sent_sent.csv",
    "config_file": "llama2_zs_hagrid_answer_GTR_10_passage_sent_sent.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}

exp_llama2_answer_RTG_gen_queries = {
    "model_name": "Llama-2-7b-chat-hf",
    "model_id": "meta-llama/Llama-2-7b-chat-hf",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "use_retrieved": True,
    "retrieved_passages_file": f"{ROOT_PATH}/results/retrieval/miracl_dataset/generated_queries_4shot_4q_lbre_nb_example_pmpt3_desc_rerank.csv",  # devMiracl_results_MonoT5_BM500_20_normal_corpus.csv",  # generated_queries_4shot_4q_lbre_nb_example_pmpt2_desc_seperate
    "use_context": True,
    "SEED": 42,
    "dataset": "HAGRID",
    "nb_passages": 10,
    "citation": True,
    "experiment_name": "RTG_generated_queries",  # "llama2_zs_hagrid_ctxt_citing",  # "llama2_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/llms/ciKM_generation_experiments/llama2_zs_hagrid_answer_gen/",
    "results_file": "llama2_zs_answer_generation_RTG_gen_queries_4q_4shots_pmpt3_rerank_10_passages.csv",
    "config_file": "llama2_zs_answer_generation_RTG_gen_queries_4q_4shots_pmpt3_rerank_10_passages.json",
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
}

CONFIG: Dict = {
    "experiment": exp_llama2_answer_GTR,  # RL_reward_answergen_exp,  # exp_zephyr_answer_without_citation,  # exp_zephyr_query_gen_fewshots_test,  # exp_zephyr_query_gen_fewshots,  # exp_zephyr_answer_without_citation,  # exp_zephyr_answer_with_retrieved_passages,  # experiment,
    "prompts": prompts_config,
    "hagrid_miracl": hagrid_miracl_config,
}
