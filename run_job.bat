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
cd /home/djeddal/Documents/Code/Attributed-IR


#python scripts/retrieve_posthoc_gtr.py
#python scripts/generate_answer.py --model_name zephyr --architcture RTG-vanilla 
python scripts/citations_eval.py --architcture RTG-vanilla --autoais Cit --results_file /home/djeddal/Documents/Code/Attributed-IR/results/G/answer_generation__GTR_adjusted.json
#python scripts/evaluate_correstness.py --architcture RTG-vanilla --multiple_gold_answers True --results_file /home/djeddal/Documents/Code/Attributed-IR/results/RTG_gold/answer_generation_RTG_gold_passages.csv
#python scripts/citations_eval_gtr.py --architcture RTG-vanilla --results_file /home/djeddal/Documents/Code/Attributed-IR/results/G/answer_generation__GTR.csv