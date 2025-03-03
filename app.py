import base64
import json
import os

import google.auth
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_google_vertexai import ChatVertexAI
from mongo_client import get_thread_source_transcription_in_json

load_dotenv()


def base64ToString(b):
    return base64.b64decode(b).decode("utf-8")


AI_NOTES_EDU_PROMPT = """
You are a students who is very good at taking concise, pointwise notes for readers from the transcript of the video. 
The notes that you take only contains important key definations,concepts, example, formulaes, Tables, mnemonics etc. 
You want to create that is easier to revise whenever needed.
The timestamp in millisecond is also attached in the trasncript, enclose the each millisecond timestamp inside double square brackets so that for readers it becomes easier to go to the video and understand the concept. Example:=>[[234]] OR [[234]]-[[2045]]
and always end your notes with small Q&A.
Start Your Notes With "Notes is provided below".
refrain from mentioning Ads, sponsers and all in the notes. We only need important information
"""

AI_NOTES_POD_PROMPT = """
You are someone who writes show notes for the podcast.
You will be provided with the transcript of the podcast create a beautiful concise show notes for it. 
You can include the following:
    Small Guest information (if Any).
    Organize your show notes in a logical and easy-to-follow forma. Use bullet points, headers, and short paragraphs to break up the text.
    Include any framework, personal tricks they used or talk about.
    Clearly outline the main points and topics discussed in the episode. This might include important arguments, insights, quotes.
    Link to any articles, books, websites, or products mentioned in the episode. 
    The timestamp in millisecond is also attached in the trasncript, enclose the each millisecond timestamp inside double square brackets so that for readers it becomes easier to go to the video and understand the concept. Example:=>[[234]] OR [[234]]-[[2045]]
    always end your notes with small Q&A.
Start Your Notes With "Below is Your Lovely Notes"
refrain from mentioning Ads, sponsers and all in the notes. We only need important information, not any redundant informaiton.
"""
IS_PODCAST = """
You will be provided with a transcript of a video. Your task is to determine whether the summary belongs to a podcast or a webinar. If the summary describes content that fits the format of a podcast or a webinar, return 'Yes'. Otherwise, return 'No'. Ensure that your response strictly contains only 'Yes' or 'No' without any additional text.
"""

credentials, project_id = google.auth.load_credentials_from_dict(
    json.loads(base64ToString(os.getenv("GOOGLE_CREDENTIALS")))
)
# search = GoogleSearchAPIWrapper(google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"))
llm = ChatVertexAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    max_tokens=None,
    max_retries=2,
    stop=None,
    credentials=credentials,
    project=project_id,
)


def ai_notes(video_id):
    transcript = get_thread_source_transcription_in_json(video_id)
    data = "\n".join([f"{data['offset']} {data['text']}" for data in transcript])
    final_prompt = IS_PODCAST + f"\n---\nTRANSCRIPT DATA:\n{data}"
    response = llm.invoke(final_prompt)
    if response == "Yes":
        prompt = AI_NOTES_POD_PROMPT
    else:
        prompt = AI_NOTES_EDU_PROMPT

    prompt = ChatPromptTemplate.from_messages(
        [("system", prompt), ("user", "## INPUT:\n{input}")]
    )
    chain = prompt | llm

    content = ""
    for data in chain.stream({"input": data}):
        content += data.content + " "
        yield f"{data.content}"


def main():
    st.title("AI Notes for YouTube Video")
    video_id = st.text_input("Enter YouTube Video ID:")

    if st.button("Generate Notes"):
        if not video_id.strip():
            st.error("Please enter a valid YouTube video ID.")
        else:
            # Create a placeholder for streaming output
            output_placeholder = st.empty()
            output_text = ""
            # Iterate over the streamed tokens and update the placeholder in real time
            for token in ai_notes(video_id):
                output_text += token
                output_placeholder.markdown(output_text)


if __name__ == "__main__":
    main()

# def top5_results(query):
#     return search.results(query, 5)


# def get_Search_result(question: str):
#     tool = Tool(
#         name="google_search",
#         description="Search Google for recent results.",
#         func=search.results,
#     )
#     result = tool.invoke(question)
#     return result


# st.title("Search On Steroids")

# user_question = st.text_input("Enter your question:")

# if user_question:
#     st.write("You asked:", user_question)
#     response_area = st.empty()
#     full_response = ""
#     with st.spinner("Fetching answer..."):
#         try:
#             result = top5_results(user_question)
#             for chunk in llm.stream(
#                 f"{result}\n\n---\n\n COntext is Provided Above. Answer the below quesion at best of your ability and attach any inline citation if availaible:\n\n {user_question}"
#             ):
#                 chunk_content = chunk.content
#                 full_response += chunk_content
#                 response_area.markdown(full_response)

#         except Exception as e:
#             st.error(f"An error occurred: {e}")
