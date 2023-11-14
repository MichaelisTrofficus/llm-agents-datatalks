import requests

from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import Chroma


def make_vectorstore(url_list, text_splitter, embeddings):
    loader = UnstructuredURLLoader(urls=url_list)
    web_doc = loader.load()
    web_docs = text_splitter.split_documents(web_doc)
    vectorstore = Chroma.from_documents(
        web_docs,
        embeddings,
        collection_name="web_docs")

    return vectorstore


def make_qa_retriever(llm, vectorstore):
    retriever = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False)

    return retriever


def make_sep_retriever(llm, text_splitter, embeddings, query_list):
    """build retriever for stanford encyclopedia of philosophy entries"""
    print("philosopher: ", query_list[0])
    url_list = []
    for query in query_list:
        base_url = "https://plato.stanford.edu/search/searcher.py"
        params = {"query": query}
        response = requests.get(base_url, params=params)

        # Extract the URL of the first search result
        url = None
        for line in response.iter_lines(decode_unicode=True):
            if "Search results for" in line:
                continue
            elif line.startswith('<a href="'):
                url = line.split('"')[1]
                break

        url_list.append(url)
    print("entry url: ", url_list[0])
    vectorstore = make_vectorstore(url_list, text_splitter, embeddings)
    sep_retriever = make_qa_retriever(llm, vectorstore)

    return sep_retriever
