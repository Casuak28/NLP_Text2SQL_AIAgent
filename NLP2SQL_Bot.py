#Environment Variables
from dotenv import load_dotenv
load_dotenv() ##Loading all the environment variables from .env file

from fewshot import examples1 ##calling our own python file for few shot prompting techniques
from dynamic_tables import table_details, Table

#Gemini + Streamlit
import streamlit as st
import google.generativeai as genai
import os
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI

#Langchain + OpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate


from langchain_google_genai import GoogleGenerativeAIEmbeddings         ##Vector Embeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma

#Dynamic Table Generation
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
#import pandas as pd  ##Do not activate -> creating issues with SemanticSimilarityExampleSelector

#Message history
#from langchain.memory import ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


##Configuring our Gemini API
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")


##Establishing Connection with DB

##Define the URI for the SQLite database
database_uri = "sqlite:///interview.db"

##Connect to the SQLite database using the from_uri() function
db = SQLDatabase.from_uri(database_uri)

##db object to interact with your SQLite database
#print("Connected to the database:", db)
#print(db.dialect)
#print(db.get_usable_table_names())
#print(db.table_info("Education"))  #JSON creating a problem

#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

#chains to generate Query and Execute Query
#generate_query = create_sql_query_chain(llm, db)
#execute_query = QuerySQLDataBaseTool(db=db)

#query = generate_query.invoke({"question": "Give me names of user who have experience in skill 'AWS'? Also, the sql code should not have ''' in the beginning or the end and sql word in the output."})
#print(query)
#execute_query = QuerySQLDataBaseTool(db=db)
#result = execute_query.invoke(query)

##Prompts

#Rephrasing the query output
answer_prompt = PromptTemplate.from_template("""
Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: 
"""
) ## We can add a follow up question here.


##Adding few shot examples prompt
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("user", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples1,
    input_variables=["input","top_k"],
    #input_variables=["input"],
)
#print(few_shot_prompt.format(input="How many products are there?", top_k=3))

#print("Vector")

##VectorDB and Similarity search code
vectorstore = Chroma()
# Clear or reinitialize the chromaDB vector store
vectorstore.delete_collection() 


# Initialize the GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#print("embeddings")

# Create the example selector using the new embeddings and ChromaDB vector store
try:
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples1,
        embeddings,
        vectorstore,
        k=3,
        input_keys=["input"],
    )
except Exception as e:
    print(f"An error occurred: {e}")
#print(example_selector.select_examples({"input": "Give me names of user who have experience in skill 'AWS'?"}))


##FewShotPrompt to be added to the final prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["input","top_k"],
)
#print(few_shot_prompt.format(input="Give me names of user who have experience in skill 'AWS'?"))

#Creating the final combined prompt
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("user", "You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Unless otherwise specified. You can order the results to return the most informative data in the database. You must query only the columns that are needed to answer the question. Also, the final SQLite query should not have ''' in the beginning or the end and sql word in the output.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."),
        few_shot_prompt,
        MessagesPlaceholder(variable_name = "messages"),
        ("user", "{input}"),
    ]
)
#print(final_prompt.format(input="How many products are there?",table_info="some table info"))
#print("Chains")


#Prompt for dynamic table selection
table_details_prompt = PromptTemplate.from_template("""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. The tables are:

{table_details}

Also, below is a list of past conversations which happened before user asked current question. These can be used along with user question to identify ALL of the SQL tables that MIGHT be relevant.
ONLY use them if you feel they are relevant to the current question. The past conversations are:
{messages}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed. Return the tables in the form of a python list of strings of table names.
Also, the final output should not have ''' in the beginning or the end and python word in the output.\
User Question: {question}
""")

#converting above output of string to list
def convert_string_to_list(string):
    # Remove the square brackets
    string = string.strip('[]')
    # Split the string by commas
    items = string.split(',')
    # Strip whitespace and quotes from each item
    result_list = [item.strip().strip("'") for item in items]
    return result_list

#table_chain = create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)

table_chain = table_details_prompt | llm | StrOutputParser() | convert_string_to_list
#tables = table_chain.invoke({"question": "Give me names of user who have experience in skill 'AWS'?", "table_details":table_details})
#print(tables)
#print(type(tables))


#Chains to generate the query and execute it.
generate_query = create_sql_query_chain(llm, db,final_prompt)
execute_query = QuerySQLDataBaseTool(db=db)

#Chain for Converting the sql output into human readable output
rephrase_answer = answer_prompt | llm | StrOutputParser()

chain = (
    RunnablePassthrough.assign(table_names_to_use=table_chain) |
    RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query") | execute_query
    )
    | rephrase_answer
 )

#Printing the generic prompt
# print(chain.get_prompts()[0])
# print("\n\n")
# print(chain.get_prompts()[1])


#generating the history
#history = ChatMessageHistory() #not using this because streamlit was recreating this variable everytime it refreshed causing history to not load properly

#function to generate chat history
def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history


# #Question from user
# question1 = "Give me names of user who have experience in skill 'AWS'?"

# result = chain.invoke({"question": question1, "table_details":table_details, "messages":history.messages})
# print(result)

# history.add_user_message(question1)
# history.add_ai_message(result)

# #Second question
# question2 = "And can be hired in a budget of $20000?"

# response = chain.invoke({"question":question2, "table_details":table_details, "messages":history.messages})
# print(response)
# history.add_user_message(question2)
# history.add_ai_message(response)

# print("\n----------------\n",history.messages)

##==========================================================================
##Streamlit UI Code
##==========================================================================
st.title("Shikhar's PermiBot")

# Initialize chat history
if "messages" not in st.session_state:
    # print("Creating session state")
    st.session_state.messages = []


#Adding the initial chatmesage
with st.chat_message("assistant"):
    st.markdown("Hello there! I am PermiBot - your assistant.\nYou can ask me questions related to the database and I will try my best to answer you.")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("How may I help you today?"):
    # # Add user message to chat history
    # st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            
            if st.session_state.messages: #only running history if state messages are not empty
                ##First creating history from st.session messages
                history = create_history(st.session_state.messages)
            else:
                history = ChatMessageHistory()

            print("\n--------before_history--------\n",history.messages)

            ##Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            ##Calling our model chain and adding messages to chat history
            response = chain.invoke({"question":prompt, "table_details":table_details, "messages":history.messages})
            
            ##Adding response to history
            history.add_user_message(prompt)
            history.add_ai_message(response)

            print("\n--------after_history--------\n",history.messages)

            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})