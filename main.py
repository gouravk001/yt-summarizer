from datetime import datetime
from xml.parsers.expat import model
from youtube_transcript_api import (YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound)
from urllib.parse import urlparse, parse_qs
from openai import OpenAI
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=api_key)

mongoclient = MongoClient(os.getenv("DATABASE_URL"))
db = mongoclient["yt_summarizer"]
collection = db["summaries"]

def getId(url):
    try:
        parsed_url = urlparse(url)
        host = parsed_url.hostname
        
        if host in ("www.youtube.com", "youtube.com"):
            query = parse_qs(parsed_url.query)
            if "v" in query:
                return query["v"][0]
            path_parts = parsed_url.path.strip("/").split("/")
            if len(path_parts) >= 2 and path_parts[0] in ("live", "embed", "shorts"):
                return path_parts[1]

        if host == "youtu.be":
            return parsed_url.path.lstrip("/")

    except Exception as e:
        print(f"Error parsing URL: {e}")
        return None

    return None

def fetch_script(video_id):
    
    
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)

        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        
        except NoTranscriptFound:
            transcript = next(iter(transcript_list))
            if transcript.language_code not in ['en', 'en-US', 'en-GB']:
                transcript = transcript.translate('en')

        
        return transcript.fetch()

    except (TranscriptsDisabled, NoTranscriptFound):
        raise ValueError("Subtitles are disabled or not found for this video.")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch transcript: {str(e)}")
    
def getSummary(url):
    vid_id = getId(url)
    if not vid_id:
        raise ValueError("Invalid YouTube URL")

    cached_doc = collection.find_one({"video_id": vid_id})
    if cached_doc:
        return {"summary": cached_doc["output_text"], "cached": True}

    transcript = fetch_script(vid_id)
    script = " ".join([snippet.text for snippet in transcript.snippets])

    prompt = f'You have to create a summary of this youtube video by using its transcript in 300-400 words in bullet points and with descriptive headings with proper structure and only write the summary \ntranscript = "{script}"'

    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt,
        temperature=1.0,
        top_p=1.0
    )

    print(response)
    
    summary_text = response.output_text

    collection.insert_one({
        "video_id": vid_id,
        "output_text": summary_text,
    })

    return {"summary": summary_text, "cached": False}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return {"message" : "First Data"}

@app.get("/summary")
def summary(url: str):
    try:
        res = getSummary(url)
        return {"summary": res["summary"], "cached": res["cached"]}

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except RuntimeError as re:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(re)
        )
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )