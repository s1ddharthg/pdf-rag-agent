from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

evaluator_llm = Ollama(model="llama3")
evaluator_embeddings = OllamaEmbeddings(model="nomic-embed-text")

ragas_llm = LangchainLLMWrapper(evaluator_llm)
ragas_emb = LangchainEmbeddingsWrapper(evaluator_embeddings)
