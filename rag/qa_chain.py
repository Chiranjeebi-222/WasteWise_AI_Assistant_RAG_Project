from transformers import pipeline

def create_qa_chain(vector_store):
    """
    Pure RAG QA chain using:
    - FAISS retriever
    - HuggingFace FLAN-T5 pipeline
    """

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Correct pipeline (FLAN-T5 is text2text model)
    qa_pipeline = pipeline(
        task="text2text-generation",   # ✅ FIXED
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_new_tokens=256,
        do_sample=False
    )

    def answer_question(question):
        # Retrieve relevant documents
        docs = retriever.invoke(question)

        # If no docs found
        if not docs:
            return "Answer not found in the provided waste management documents."

        # Convert documents to text
        context = "\n\n".join(doc.page_content for doc in docs)

        # Create prompt
        prompt = f"""
You are a waste management assistant.
Answer ONLY using the context below.

Context:
{context}

Question: {question}
Answer:
"""

        # Generate answer
        result = qa_pipeline(prompt)

        # Return generated text
        return result[0]["generated_text"]

    return answer_question
