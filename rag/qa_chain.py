from transformers import pipeline

def create_qa_chain(vector_store):
    """
    Pure RAG QA chain using:
    - FAISS retriever
    - HuggingFace text-generation pipeline
    - NO LangChain prompt dependency
    """

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    qa_pipeline = pipeline(
        task="text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_new_tokens=256,
        do_sample=False
    )

    def answer_question(question: str) -> str:
        docs = retriever.get_relevant_documents(question)

        if not docs:
            return "Answer not found in the provided waste management documents."

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = (
            "You are a waste management assistant.\n"
            "Answer ONLY using the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

        result = qa_pipeline(prompt)
        return result[0]["generated_text"]

    return answer_question
