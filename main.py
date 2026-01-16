from datetime import datetime
from urllib.parse import urlparse, parse_qs
import os
import requests
from dotenv import load_dotenv

from openai import OpenAI
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key not found in environment (OPEN_API_KEY or OPENAI_API_KEY)")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

MONGO_URI = os.getenv("DATABASE_URL")
if not MONGO_URI:
    raise RuntimeError("DATABASE_URL not set in environment")

mongoclient = MongoClient(MONGO_URI)
db = mongoclient["yt_summarizer"]
collection = db["summaries"]
try:
    collection.create_index("video_id", unique=True)
except Exception:
    pass

# Transcript API
TRANSCRIPT_API_KEY = os.getenv("YOUTUBE_TRANSCRIPT_API_KEY")
TRANSCRIPT_API_URL = os.getenv("YOUTUBE_TRANSCRIPT_API_URL") or "https://www.youtube-transcript.io/api/transcripts"
if not TRANSCRIPT_API_KEY:
    pass


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)


def getId(url: str) -> str | None:
    """Extract youtube video id from various URL forms."""
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


def normalize_transcript(transcript) -> str:
   
    if transcript is None:
        raise ValueError("Transcript is empty")

    if isinstance(transcript, str):
        return transcript.strip()

    if isinstance(transcript, list):
        parts = []
        for item in transcript:
            
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text") or item.get("caption") or ""
                if text:
                    parts.append(text.strip())
        return " ".join(parts).strip()

    if isinstance(transcript, dict):
        if "transcript" in transcript:
            return normalize_transcript(transcript["transcript"])
        if "tracks" in transcript:
            tracks = transcript["tracks"]
            if isinstance(tracks, list) and len(tracks) > 0:
                first = tracks[0]
                if isinstance(first, dict) and "transcript" in first:
                    return normalize_transcript(first["transcript"])
      
        texts = []
        for v in transcript.values():
            if isinstance(v, str):
                texts.append(v.strip())
            elif isinstance(v, list):
                texts.append(normalize_transcript(v))
        joined = " ".join([t for t in texts if t])
        if joined:
            return joined.strip()

    raise ValueError("Unsupported transcript format")


def fetch_script(video_id: str) -> str:
  
    if not TRANSCRIPT_API_KEY:
        raise RuntimeError("Transcript API key not configured in environment (YOUTUBE_TRANSCRIPT_API_KEY)")

    headers = {
        "Authorization": f"Basic {TRANSCRIPT_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"ids": [video_id]}

    try:
        resp = requests.post(TRANSCRIPT_API_URL, headers=headers, json=payload, timeout=200)
    except requests.exceptions.Timeout:
        raise RuntimeError("Transcript API request timed out")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Transcript API request failed: {str(e)}")

    if resp.status_code != 200:
        raise RuntimeError(f"Transcript API error {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError("Transcript API returned invalid JSON")

  
    video_obj = None
    if isinstance(data, list) and len(data) > 0:
        video_obj = data[0]
    elif isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list) and len(data["results"]) > 0:
            video_obj = data["results"][0]
        else:
            video_obj = data

    if not video_obj:
        raise ValueError("Transcript not found in API response")
    raw_transcript = None
    for key in ("transcript", "text", "tracks"):
        if key in video_obj and video_obj[key]:
            raw_transcript = video_obj[key]
            break

    if raw_transcript is None:
        tracks = video_obj.get("tracks") or video_obj.get("captions")
        if isinstance(tracks, list) and len(tracks) > 0:
            t0 = tracks[0]
            if isinstance(t0, dict) and "transcript" in t0:
                raw_transcript = t0["transcript"]

    if raw_transcript is None:
        if isinstance(video_obj, dict) and "text" in video_obj and video_obj["text"]:
            raw_transcript = video_obj["text"]

    if raw_transcript is None:
        raise ValueError("Transcript not present in API response for video_id=" + video_id)

    script_text = normalize_transcript(raw_transcript)
    if not script_text:
        raise ValueError("Transcript returned but empty after normalization")

    return script_text


def getSummary(url: str) -> dict:
  
    vid_id = getId(url)
    if not vid_id:
        raise ValueError("Invalid YouTube URL")

    cached_doc = collection.find_one({"video_id": vid_id})
    if cached_doc and "output_text" in cached_doc:
        return {"summary": cached_doc["output_text"], "cached": True}

    script = fetch_script(vid_id)

    MAX_CHARS = 20000
    if len(script) > MAX_CHARS:
        script = script[:MAX_CHARS]

    prompt = (
        "You have to create a summary of this youtube video by using its transcript in 300-400 words "
        "in bullet points and with descriptive headings with proper structure and only write the summary.\n\n"
        "Transcript:\n"
        f"{script}"
    )

    try:
        response = openai_client.responses.create(
            model="gpt-5-nano",
            input=prompt,
            temperature=1.0,
            top_p=1.0
        )
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {str(e)}")

    summary_text = None
    if hasattr(response, "output_text"):
        summary_text = response.output_text
    else:
        
        try:
            out = getattr(response, "output", None) or response.get("output") if isinstance(response, dict) else None
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict):
                    if "content" in first and isinstance(first["content"], list):
                        for c in first["content"]:
                            if isinstance(c, dict) and "text" in c:
                                summary_text = c["text"]
                                break
                    summary_text = summary_text or first.get("text") or first.get("output_text")
            if not summary_text and isinstance(response, dict):
                summary_text = response.get("output_text") or response.get("text")
        except Exception:
            summary_text = None

    if not summary_text:
        raise RuntimeError("Unable to extract summary text from LLM response")

    try:
        collection.insert_one({
            "video_id": vid_id,
            "output_text": summary_text,
            "created_at": datetime.utcnow()
        })
    except Exception:
        print("Warning: failed to write summary to MongoDB for video_id=", vid_id)

    return {"summary": summary_text, "cached": False}



@app.get("/")
def index():
    return {"message": "First Data"}


@app.get("/summary")
def summary(url: str):
    try:
        res = getSummary(url)
        return {"summary": res["summary"], "cached": res["cached"]}

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except RuntimeError as re:
        print("Runtime error:", re)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(re))
    except Exception as e:
        print("Unexpected error:", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An unexpected error occurred: {str(e)}")
