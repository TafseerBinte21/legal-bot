from langchain_community.llms import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
from langchain import PromptTemplate
from embedding import docsearch
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory


# Define the repo ID and connect to Mixtral model on Huggingface
llm = ChatOpenAI(model="gpt-3.5-turbo")


template = """
You are an expert on Bangladeshi laws, particularly those related to women and children. 
These Humans will ask you questions about the acts, laws, and punishments based on specific incidents. 
Use the following piece of context to answer the question. Also you will always remember an user name.
If you don't know the answer, just say you don't know. 
Keep the answer concise and within 2 sentences.

Context: {context}
Question: {question}
Answer: 

"""

prompt = PromptTemplate(
  template=template, 
  input_variables=["context", "question"]
)

# Set up memory
memory = ConversationBufferMemory(memory_key="history", input_key="question")


rag_chain = (
  {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
  | prompt  
  | llm
  | StrOutputParser() 
)

# input = input("Ask me anything: ")
# result = rag_chain.invoke(input)
# print(result)


# Register user name
user_name = input("Please enter your name: ")
memory.save_context({"question": f"My name is {user_name}"}, {"answer": "Noted"})


while True:
    user_input = input("Ask me anything : ")
    if user_input.lower() == "ok bye":
        print("Conversation ended.")
        break
    result = rag_chain.invoke(user_input)
    print(result) 