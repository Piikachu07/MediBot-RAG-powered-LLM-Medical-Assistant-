import os 

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Setup LLM (Mistral with HuggingFace)

HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(Huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id= huggingface_repo_id,
        temperature= 0.5,
        model_kwargs={"token": HF_TOKEN,
                      "max_length": "600"}

    )
    return llm

# Connect LLM with FAISS(datastore i.e memory) and Create chain

DB_FAISS_PATH = "vectorstore/df_faiss"
custom_prompt_template = """

Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, don't try to make up an answer.
Don't provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. Little bit small talk allowed but again dont give false information be accurate.
"""


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template= custom_prompt_template, input_variables= ["context", "question"])
    return prompt

DB_FAISS_PATH = "vectorstore/df_faiss"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model,  allow_dangerous_deserialization= True)

#QA_chain

qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(huggingface_repo_id),
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs = {"k": 2}),  # top kitna documents similar chaiye to return in answer
    return_source_documents = True,
    chain_type_kwargs = {"prompt": set_custom_prompt(custom_prompt_template)}
)


# Now invoke with a single query

user_query = input("Write Your Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("Result: ", response["result"])
print("SOURCE DOCUMENTSL: ", response['source_documents'])
