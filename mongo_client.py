from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
db_client = None
db_name = os.getenv("MONGO_DB_NAME")
db_async_client = None


def get_db():
    global db_client
    if db_client is None:
        db_client = MongoClient(os.getenv("MONGO_DB_URI"))
    return db_client[db_name]


def get_thread_source_transcription_in_json(video_id: str):
    try:
        db = get_db()
        thread_source_collection = db["thread_source_datas"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        thread_source_doc = thread_source_collection.find_one({"thread_url": url})

        if not thread_source_doc:
            raise ValueError(f"Document with id {video_id} not found")

        if (
            thread_source_doc["youtube_metadata"] is None
            or thread_source_doc["youtube_metadata"]["transcriptions"] is None
            or len(thread_source_doc["youtube_metadata"]["transcriptions"]) == 0
        ):
            raise ValueError(
                f"Document with id {video_id} does not have transcriptions"
            )

        json_data = thread_source_doc["youtube_metadata"]["transcriptions"][0][
            "transcription"
        ]
        chapters = thread_source_doc["youtube_metadata"]["chapters"]
        if chapters:
            chapters = [chapter["title"] for chapter in chapters]
        for dictionary in json_data:
            del dictionary["_id"]
        return json_data, chapters

    except Exception as e:
        print(e)
        raise ValueError(
            f"Some error occured while fetching transcription for document {video_id}"
        )
