from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def get_rag_chain(llm, vector_db):
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    template = """
    Instruction: Use the context to provide a direct 5-sentence answer. 
    You are a direct and concise assistant. 
    Do not mention 'the context' or 'the documents'. 
    Do not add labels like 'Assistant:' or notes.

    Context: {context}
    Question: {question}
    
    Final Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
