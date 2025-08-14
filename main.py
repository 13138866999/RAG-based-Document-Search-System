from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory="./db", embedding_function=embedding_function)
inputText = "How does Alice meet the Mad Hatter"
res = db.similarity_search_with_relevance_scores(inputText, k = 3)

#check if the res is enough and score is enough
if len(res) == 0 or res[0][1] < 0.7:
    print("No similarity scores found")


context = "\n\n------\n\n".join([doc.page_content for doc, _scores in res])
print(context)

inputTemplate = (""
                 "Below is the context your answer should be based on"
                 "\n\n"
                 +context
                 +"\n\n"
                 "answer this question based on above information:"
                  "How does Alice meet the Mad Hatter"
                 )

model = ChatOpenAI()

response = model.predict(inputTemplate)
print(response)