#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=eval # le nom du job (voir commande squeue)
#SBATCH --nodes=1 # le nombre de noeuds
#SBATCH --gpus=1 # nombre de gpu
#SBATCH --ntasks-per-node=1 # nombre de tache par noeud 
#SBATCH --time=1-90:00:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=jz_%j_%x.out     # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out      # nom du fichier d'erreur (ici commun avec la sortie)

# Source l'environement par example ~/.bashrc
source ~/.bashrc
# activer l'environement python
conda activate llms-env
cd ....


#python scripts/generate_answer.py --model_name zephyr --architecture G

#python scripts/generate_answer.py --model_name zephyr --architecture RTG-gold

#python scripts/retrieve.py 
#python scripts/generate_answer.py --model_name zephyr --architecture RTG-vanilla 


#python scripts/generate_queries.py 
#python scripts/retrieve_with_generated_queries.py
#python scripts/generate_answer.py --model_name zephyr --architecture RTG-query-gen


#python scripts/retrieve_posthoc_gtr.py
#python scripts/citations_eval.py --architecture RTG-vanilla --autoais Cit 
python scripts/citations_eval.py --architecture RTG-gold --autoais ALCE --overlap 
#python scripts/evaluate_correstness.py --architecture RTG-vanilla --multiple_gold_answers --results_file ...
#python scripts/citations_eval_gtr.py --architecture RTG-vanilla --results_file ...