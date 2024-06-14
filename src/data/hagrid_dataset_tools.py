import datasets
import re


def load_dataset(split="train"):
    hagrid = datasets.load_dataset("miracl/hagrid", split=split)
    return hagrid


def prepare_contexts(context_list, hagrid_gold=False, citation=True):
    """
    concatenate the contexts to use as prompt
    """
    if hagrid_gold:
        if citation:
            offset = 0
            if context_list[0]["idx"] == 0:
                offset = 1
            context_text = [
                "[" + str((ctxt["idx"] + offset)) + "] " + ctxt["text"]
                for ctxt in context_list
            ]
        else:
            context_text = [ctxt["text"] for ctxt in context_list]
    else:
        if citation:
            context_text = [
                "[" + str((i + 1)) + "] " + context_list[i]["text"]
                for i in range(len(context_list))
            ]
        else:
            if isinstance(context_list[0], dict):
                context_text = [ context_list[i]["text"] for i in range(len(context_list))]
            else:
                context_text = context_list




        
    return "\n".join(context_text)


def get_attributable_answer(answers, text_only=True):
    """
    For each entry, returns the first attributable answer.

    text_only : if True returns the full text of the answer. Else returns the answer split into sentences
    """
    att_and_info = []
    att_only = []
    info_only = []

    for answer in answers:
        if answer["attributable"] == 1:
            if answer["informative"] == 1:
                att_and_info.append(answer)
            else:
                att_only.append(answer)
        else:
            if answer["informative"] == 1:
                info_only.append(answer)
    if len(att_and_info):
        final_answer = att_and_info[0]
    elif len(att_only):
        final_answer = att_only[0]
    elif len(info_only):
        final_answer = info_only[0]
    else:
        final_answer = answers[0]

    if text_only:
        return final_answer["answer"]
    return final_answer


def get_all_answers(
    answers, text_only=True, with_citiations=False
):  # fonction de raouf
    """
    fonction return all the reference answers (type list) for each answers

    text_only : if True we return only the reference answers text else all structure
    with_citiations : if False we remove citiations from reference answers text else we don't

    """

    expression = r"\[\d+(?:,\s*\d+)*\]"

    if text_only:
        if not with_citiations:
            return [re.sub(expression, "", s["answer"]) for s in answers]
        else:
            return [s["answer"] for s in answers]
    else:
        if not with_citiations:
            for s in answers:
                s["answer"] = re.sub(expression, "", s["answer"])

        return answers


def get_non_attributable_answers(df):
    """
    Analysis answers in Hagrid. Returns list of queries with no attributable answers (not even attributed sentences), etc
    """
    not_attribuatble_queries = []
    no_attributable_sent_queries = []
    not_informative_queries = []
    no_informative_sent_queries = []
    for _, row in df.iterrows():
        answers = row["gold_truth"]
        not_attribuatble = True
        no_attributable_sent = True
        not_informative = True
        no_informative_sent = True
        for answer in answers:
            if answer["attributable"] == 1:
                not_attribuatble = False
            if answer["informative"] == 1:
                not_informative = False
            for sent in answer["sentences"]:
                if sent["attributable"] == 1:
                    no_attributable_sent = False
                if sent["informative"] == 1:
                    no_informative_sent = False

        if not_attribuatble == True:
            not_attribuatble_queries.append(row["query"])
        if no_attributable_sent == True:
            no_attributable_sent_queries.append(row["query"])
        if not_informative == True:
            not_informative_queries.append(row["query"])
        if no_informative_sent == True:
            no_informative_sent_queries.append(row["query"])
    return (
        not_attribuatble_queries,
        no_attributable_sent_queries,
        not_informative_queries,
        no_informative_sent_queries,
    )
