from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from . import store
from . import loader
from time import perf_counter

def process_data(path, db_path):
    doc = loader.loader_md(path)
    chunks = loader.split_to_chunk(doc)
    store.dataset_creation(chunks, db_path)


class ChromaData:
    def __init__(self):
        self.db = None

    def load_data(self, source):
        embedding_function = OpenAIEmbeddings()
        self.db = Chroma(
            persist_directory=str(source),
            embedding_function=embedding_function
        )


    def ask(self, user_question, k = 5, threshold = 0.4):
        if self.db is None:
            return None
        t0 = perf_counter()

        res = self.db.similarity_search_with_relevance_scores(user_question, k = k)
        ls_for_data = []
        print(res)
        for doc, score in res:
            snippet = doc.page_content
            meta = doc.metadata
            ls_for_data.append(dict(content=snippet, score=float(score), source=meta.get('source'),
                                    chunk=meta.get('start_index')))
        #check if the res is enough and score is enough
        if len(res) == 0 or res[0][1] < threshold:
            return dict(answer="Not enough evidence in the docs to answer confidently",
                        sources=ls_for_data[:3], latency_s = perf_counter() - t0)

        context = "\n\n------\n\n".join([doc.page_content for doc, _scores in res])
        print(context)

        input_template = (""
                         "Below is the context your answer should be based on"
                         "\n\n"
                         +context
                         +"\n\n"
                         "answer this question based on above information:"
                         + user_question
                         )

        model = ChatOpenAI()

        response = model.predict(input_template)
        return dict(answer=response, sources=ls_for_data, latency_s = perf_counter() - t0)

if __name__ == "__main__":
    chroma = ChromaData()
    chroma.load_data("../db")

    print(chroma.ask(user_question="Is Alice a woman or man?"))
    #process_data("../data", "../db")