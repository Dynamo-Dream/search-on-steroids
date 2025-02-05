import os
import base64 
import json
import google.auth
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_vertexai import ChatVertexAI
import streamlit as st
def base64ToString(b):
    return base64.b64decode(b).decode('utf-8')

credentials,project_id = google.auth.load_credentials_from_dict(json.loads(base64ToString(os.getenv("GOOGLE_CREDENTIALS"))))
search = GoogleSearchAPIWrapper(google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"))
llm = ChatVertexAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    max_tokens=None,
    max_retries=2,
    stop=None, credentials = credentials,project=project_id
)
def top5_results(query):
    return search.results(query, 5)

def get_Search_result(question:str):
    tool = Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=search.results,
    )
    result = tool.invoke(question)
    return result



st.title("Search On Steroids")

user_question = st.text_input("Enter your question:")

if user_question:
    st.write("You asked:", user_question)
    response_area = st.empty()
    full_response = ""
    with st.spinner("Fetching answer..."):
        try:
            result = top5_results(user_question)
            for chunk in llm.stream(f"{result}\n\n---\n\n COntext is Provided Above. Answer the below quesion at best of your ability and attach any inline citation if availaible:\n\n {user_question}"):
                chunk_content = chunk.content
                full_response += chunk_content
                response_area.markdown(full_response)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
