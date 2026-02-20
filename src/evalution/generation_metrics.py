import os
import sys
import evaluate
import datasets
import numpy as np

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)

current_module = sys.modules[__name__]


def rouge_detailed_ov(generated, reference, CONFIG , **kwargs):
    """
    Huggingface old version of rouge that details precision/recall/fscore
    """
    metric = datasets.load_metric(
        "rouge", cache_dir=CONFIG["evaluation"]["cache_dir"] + "rouge_old_version/",trust_remote_code=True
    )
    metric.add_batch(predictions=generated, references=reference)
    results = metric.compute()
    return results

def rouge_detailed_ov_all(generated, reference, CONFIG , **kwargs):
    """
    Huggingface old version of rouge that details precision/recall/fscore

    we calcul the rouge for all the refrence answer for each output and choose the best one
    """

    metric = datasets.load_metric(
        "rouge", cache_dir=CONFIG["evaluation"]["cache_dir"] + "rouge_old_version/", trust_remote_code=True
    )
    
    result = {'precision' : [], 'recall' : [], 'fmeasure': [] }

    for pred, refs in zip(generated, reference):
        # cree un batch 
        batch_pred = [pred] * len(refs)
        batch_ref = refs

        # calcul du rouge
        results_stochastique = metric.compute(predictions = batch_pred, references = batch_ref, use_aggregator = False)

        #add de rouge score
        best_one = sorted(results_stochastique['rougeLsum'], key=lambda x: x[2], reverse=True)[0]
        result["precision"].append(best_one[0])
        result["recall"].append(best_one[1])
        result["fmeasure"].append(best_one[2])
            
    return result


def rouge(generated, reference,CONFIG , rouge_types=["rougeLsum"], **kwargs):
    """
    New Huggingface version of rouge which returns one aggregated value
    """
    metric = evaluate.load("rouge", cache_dir=CONFIG["evaluation"]["cache_dir"], trust_remote_code=True)
    metric.add_batch(predictions=generated, references=reference)
    results = metric.compute(rouge_types=rouge_types)
    return results


def exact_match(generated, reference,CONFIG ,**kwargs):
    metric = evaluate.load("exact_match", cache_dir=CONFIG["evaluation"]["cache_dir"])
    metric.add_batch(predictions=generated, references=reference)
    results = metric.compute()
    return results


def bleu(generated, reference,CONFIG ,**kwargs):
    metric = evaluate.load("bleu", cache_dir=CONFIG["evaluation"]["cache_dir"])
    metric.add_batch(predictions=generated, references=[[r] for r in reference])
    results = metric.compute()
    return results

def bleu_all_agg(generated, reference,CONFIG ,**kwargs):
    metric = evaluate.load("bleu", cache_dir=CONFIG["evaluation"]["cache_dir"])
    results = metric.compute(predictions = generated, references = reference)
    return results

def bleu_all(generated, reference, CONFIG, **kwargs):
    metric = evaluate.load("bleu", cache_dir=CONFIG["evaluation"]["cache_dir"])
    
    # Store individual scores
    bleu_scores = []
    
    for pred, refs in zip(generated, reference):
        # If prediction is empty (no tokens) then BLEU would lead to a brevity-penalty division-by-zero.
        if not isinstance(pred, str) or len(pred.strip().split()) == 0:
            bleu_scores.append(0.0)
            continue

        # Compute BLEU for this single example (refs should be a list)
        batch_pred = [pred] * len(refs)
        batch_ref = refs
        result = metric.compute(predictions=batch_pred, references=batch_ref)
        bleu_scores.append(result['bleu'])
    
    # Also compute aggregate statistics
    overall_result = metric.compute(predictions=generated, references=reference)
    
    return {
        'bleu': overall_result['bleu'],  # Overall BLEU score
        'bleu_per_example': bleu_scores,  # List of individual scores
        'precisions': overall_result.get('precisions', [0.0,0.0,0.0,0.0]),
        'brevity_penalty': overall_result.get('brevity_penalty', 0.0),
        'length_ratio': overall_result.get('length_ratio', 0.0),
    }

def bert_score(generated, reference,CONFIG ,lang="en", **kwargs):
    metric = evaluate.load("bertscore", cache_dir=CONFIG["evaluation"]["cache_dir"])
    metric.add_batch(predictions=generated, references=reference)
    results = metric.compute(lang=lang, verbose=True)
    return results

def bert_metric_all(generated, reference,CONFIG, lang="en", **kwargs):
    metric = evaluate.load("bertscore", cache_dir=CONFIG["evaluation"]["cache_dir"])

    result = {'precision' : [], 'recall' : [], 'f1': [] }

    for pred, refs in zip(generated, reference):
        # cree un batch 
        batch_pred = [pred] * len(refs)
        batch_ref = refs

        # calcul du bert
        results_stochastique = metric.compute(predictions = batch_pred, references = batch_ref, lang=lang,verbose = True)

        #add the bert score
        best_index = np.argmax(results_stochastique['f1'])
        result['precision'].append(results_stochastique['precision'][best_index])
        result['recall'].append(results_stochastique['recall'][best_index])
        result['f1'].append(results_stochastique['f1'][best_index])
                
    return result


def meteor(generated, reference,CONFIG ,**kwargs):
    metric = evaluate.load("meteor", cache_dir=CONFIG["evaluation"]["cache_dir"])
    metric.add_batch(predictions=generated, references=reference)
    results = metric.compute()
    return results


def evaluate_metrics(generated, references, *metrics, **kwargs):
    results = {
        metric: getattr(current_module, metric, kwargs)(generated, references)
        for metric in metrics
    }

    return results
