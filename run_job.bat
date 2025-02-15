#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=selfraghagrid # le nom du job (voir commande squeue)
#SBATCH --nodelist=lizzy
#SBATCH --nodes=1 # le nombre de noeuds
#SBATCH --gpus=1 # nombre de gpu
#SBATCH --ntasks-per-node=1 # nombre de tache par noeud 
#SBATCH --time=1-90:00:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=jz_%j_%x.out     # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out      # nom du fichier d'erreur (ici commun avec la sortie)

# Source l'environement par example ~/.bashrc
source ~/.bashrc
# activer l'environement python
conda activate selfrag
#conda activate RagnRoll
cd /home/djeddal/Documents/Code/Attributed-IR/


#python scripts/generate_answer.py --model_name zephyr --architecture G

#python scripts/generate_answer.py --model_name zephyr --architecture RTG-gold

#python scripts/retrieve.py 
#python scripts/generate_answer.py --model_name zephyr --architecture RTG-vanilla 


#python scripts/generate_queries.py 
#python scripts/retrieve_with_generated_queries.py
#python scripts/generate_answer.py --model_name zephyr --architecture RTG-query-gen


#python scripts/retrieve_posthoc_gtr.py
#python scripts/citations_eval.py --architecture RTG-vanilla --autoais Cit 
#python scripts/citations_eval.py --architecture RTG-gold --autoais ALCE --overlap 
#python scripts/evaluate_correctness.py --architecture RTG-vanilla --multiple_gold_answers --results_file ...
#python scripts/citations_eval_gtr.py --architecture RTG-vanilla --results_file ...


#python scripts/evaluate_correctness.py --architecture RTG-vanilla --multiple_gold_answers --results_file /home/djeddal/Documents/Code/results/llama_7b_normal_hagridTrain/asqaTest/all_llama_testasqa_7b_finetunedSimpleAgenHagrid_4rounds_3docs.json
#python scripts/citations_eval.py --architecture RTG-vanilla --autoais Cit --results_file /home/djeddal/Documents/Code/results/llama_7b_normal_hagridTrain/asqaTest/all_llama_testasqa_7b_finetunedSimpleAgenHagrid_4rounds_3docs.json
#python scripts/citations_eval.py --architecture RTG-gold --autoais ALCE --results_file /home/djeddal/Documents/Code/results/llama_7b_normal_hagridTrain/asqaTest/all_llama_testasqa_7b_finetunedSimpleAgenHagrid_4rounds_3docs.json




#python scripts/generate_answer.py  --architecture G --model_name llama --model_id meta-llama/Llama-2-13b-chat-hf

#python scripts/generate_answer.py --architecture RTG-gold --model_name llama --model_id meta-llama/Llama-2-13b-chat-hf

#python scripts/retrieve.py 
#python scripts/generate_answer.py --architecture RTG-vanilla --model_name llama --model_id meta-llama/Llama-2-13b-chat-hf


#python scripts/generate_queries.py 
#python scripts/retrieve_with_generated_queries.py
#python scripts/generate_answer.py--architecture RTG-query-gen --model_name llama --model_id meta-llama/Llama-2-13b-chat-hf

############
#python scripts/retrieve_posthoc_gtr.py

#python scripts/evaluate_correctness.py --architecture G --multiple_gold_answers --results_file /home/djeddal/Documents/Code/Attributed-IR/results/llama13/G/answer_generation_G.csv


#python scripts/evaluate_correctness.py --architecture RTG-gold --multiple_gold_answers --results_file /home/djeddal/Documents/Code/Attributed-IR/results/llama13/RTG_gold/answer_generation_RTG_gold_passages.csv
####python scripts/citations_eval.py --architecture RTG-gold --autoais Cit --results_file /home/djeddal/Documents/Code/Attributed-IR/results/llama13/RTG_gold/answer_generation_RTG_gold_passages.csv
#python scripts/citations_eval.py --architecture RTG-gold --autoais ALCE --results_file /home/djeddal/Documents/Code/Attributed-IR/results/llama13/RTG_gold/answer_generation_RTG_gold_passages.csv

#python scripts/evaluate_correctness.py --architecture RTG-vanilla --multiple_gold_answers --results_file /home/djeddal/Documents/Code/Attributed-IR/results/llama13/RTG_vanilla/generation_RTG_vanilla_2_passages.csv
#python scripts/citations_eval.py --architecture RTG-vanilla --autoais Cit --results_file /home/djeddal/Documents/Code/Attributed-IR/results/llama13/RTG_vanilla/generation_RTG_vanilla_2_passages.csv
#python scripts/citations_eval.py --architecture RTG-vanilla --autoais ALCE --results_file /home/djeddal/Documents/Code/Attributed-IR/results/llama13/RTG_vanilla/generation_RTG_vanilla_2_passages.csv

#python scripts/citations_eval.py --architecture RTG-gold --autoais ALCE --overlap 
#python scripts/citations_eval_gtr.py --architecture RTG-vanilla --results_file ...


#python scripts/evaluate_correctness.py --architecture RTG-vanilla --multiple_gold_answers --results_file /home/djeddal/Documents/Code/results/llama_7b_normal_hagridTrain/asqaTest/all_llama_testasqa_7b_finetunedSimpleAgenHagrid_4rounds_3docs.json
#python scripts/citations_eval.py --architecture RTG-vanilla --autoais Cit --results_file /home/djeddal/Documents/Code/results/llama_7b_normal_hagridTrain/asqaTest/all_llama_testasqa_7b_finetunedSimpleAgenHagrid_4rounds_3docs.json
#python scripts/citations_eval.py --architecture RTG-gold --autoais ALCE --results_file /home/djeddal/Documents/Code/results/llama_7b_normal_hagridTrain/asqaTest/all_llama_testasqa_7b_finetunedSimpleAgenHagrid_4rounds_3docs.json


######## ragnroll ALCE
#python scripts/evaluate_correctness.py --architecture RTG-vanilla --multiple_gold_answers --results_file /home/djeddal/Documents/Code/results_jz/13b_stat_att_only/all_testasqa_llama-2-chat-hagrid-att-rag-agent-13b_4rounds_3docs.json
python scripts/citations_eval.py --architecture RTG-vanilla --autoais  ALCE --results_file /home/djeddal/Documents/Code/ALCE/result/asqa-Llama-2-13b-chat-hf-gtr_light_inst-shot0-ndoc0-42.post_hoc_cite.gtr-t5-large-external.json
#python scripts/citations_eval.py --architecture RTG-vanilla --autoais Cit --results_file /home/djeddal/Documents/Code/results_jz/13b_stat_att_only/all_testasqa_llama-2-chat-hagrid-att-rag-agent-13b_4rounds_3docs.json

######## selfrag ALCE
python scripts/evaluate_correctness.py --architecture RTG-vanilla --multiple_gold_answers --results_file /home/djeddal/Documents/Code/self-rag/retrieval_lm/selfrag_hagrid_with_retrieval_5docs_13b_gtrbm25.json
python scripts/citations_eval.py --architecture RTG-vanilla --autoais Cit --results_file /home/djeddal/Documents/Code/self-rag/retrieval_lm/selfrag_hagrid_with_retrieval_5docs_13b_gtrbm25.json
python scripts/citations_eval.py --architecture RTG-vanilla --autoais ALCE --results_file /home/djeddal/Documents/Code/self-rag/retrieval_lm/selfrag_hagrid_with_retrieval_5docs_13b_gtrbm25.json

######## baselines 
#python scripts/evaluate_correctness.py --architecture RTG-vanilla --multiple_gold_answers --results_file /home/djeddal/Documents/Code/ALCE/result/asqa-Llama-2-13b-chat-hf-gtr_light_inst-shot0-ndoc0-42.post_hoc_cite.gtr-t5-large-external.json
#python scripts/citations_eval.py --architecture RTG-vanilla --autoais Cit --results_file /home/djeddal/Documents/Code/ALCE/result/asqa-Llama-2-13b-chat-hf-gtr_light_inst-shot0-ndoc0-42.post_hoc_cite.gtr-t5-large-external.json
#python scripts/citations_eval.py --architecture RTG-vanilla --autoais ALCE --results_file /home/djeddal/Documents/Code/ALCE/result/asqa-Llama-2-13b-chat-hf-gtr_light_inst-shot0-ndoc0-42.post_hoc_cite.gtr-t5-large-external.json


######## ALCE selfask 
#python scripts/evaluate_correctness.py --architecture RTG-vanilla --multiple_gold_answers --results_file /home/djeddal/Documents/Code/self-ask/alce_asqa_selfAsk_withRetrieval_originalprompt_13b.json
