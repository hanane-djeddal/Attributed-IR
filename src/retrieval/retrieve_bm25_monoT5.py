import sys
import os

from pyserini.search.lucene import LuceneSearcher


ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)

from src.models.monoT5 import MonoT5


class Retriever:
    def __init__(self, index="miracl-v1.0-en"):
        self.docs_ids = []
        self.searcher = LuceneSearcher.from_prebuilt_index(index)
        self.ranker = MonoT5(device="cuda")

    def search(self, query, k=10):
        docs = self.searcher.search(query, k=100)
        retrieved_docid = [i.docid for i in docs]
        docs_text = [
            eval(self.searcher.doc(docid).raw())
            for _, docid in enumerate(retrieved_docid)
        ]
        ranked_doc = self.ranker.rerank(query, docs_text)[:k]
        docids = [i["docid"] for i in ranked_doc]
        scores = [i["score"] for i in ranked_doc]
        docs_text = [
            eval(self.searcher.doc(docid).raw()) for j, docid in enumerate(docids)
        ]
        docs = [
            {"id": docids[i], "text": docs_text[i], "score": scores[i]}
            for i in range(len(docids))
        ]
        return docs

    def process(self, query, **kwargs):
        docs_text = self.search(query, **kwargs)
        return f"\n[DOCS] {docs_text} [/DOCS]\n"
