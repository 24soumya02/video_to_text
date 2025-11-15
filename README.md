# Automated Audio Transcription Pipeline using Gemini + FFmpeg + Cloudflare D1 + GCS

This project is an end-to-end automated pipeline that:

1. Downloads an MP4 video  
2. Converts it to MP3  
3. Chunks the audio into 5-minute segments using FFmpeg  
4. Transcribes each chunk using Gemini 2.5 Flash  
5. Merges all transcript segments with accurate timestamps  
6. Stores the output transcript (.json + .txt) in Google Cloud Storage  
7. Retrieves transcript text and prepares a structured DataFrame  
8. Stores updated tokens in Cloudflare D1 database  

This system is built for real-time, large-file processing and healthcare-specific transcription with multilingual → English translation.

## Features

### 1. MP4 Download
Downloads video content using streaming chunks to avoid memory issues.

### 2. MP4 → MP3 Conversion
Uses FFmpeg CLI for high-quality audio extraction.

### 3. Audio Chunking
Splits audio into equal 5-minute segments using FFmpeg:
- Works for any duration  
- Creates optimized MP3 chunks  
- Handles file size validations  

### 4. Gemini Audio Transcription (Base64 Method)
Each chunk:
- Is read as base64  
- Sent directly to Gemini  
- Transcribed + translated to English  
- Returned as JSON with timestamps  

### 5. JSON Transcript Merging
- Merges all chunk outputs  
- Converts timestamps to absolute time  
- Removes duplicate timestamps  
- Produces a clean chronological transcript  

### 6. Cloudflare D1 Token Rotation
- Reads the oldest Gemini token  
- Updates timestamp  
- Enables round‑robin token usage  

### 7. Upload to Google Cloud Storage
Saves:
- Final JSON transcript  
- Final TXT transcript  

Stored inside:

`gs://video_transcript/gemini_transcript/{id}.txt`

### 8. Result Packaging
Final DataFrame format:

| type | type_id | srt |
|------|---------|-----|
| 2 | content_id | transcript text |

## Technology Stack

- Python 3  
- FFmpeg / FFprobe  
- Google Generative AI (Gemini)  
- Cloudflare D1  
- Google Cloud Storage  
- Pandas, Requests, Pydub  

## Output Files

- `{id}.json` – full structured transcript  
- `{id}.txt` – readable transcript  

Uploaded to:

`https://storage.googleapis.com/video_transcript/gemini_transcript/{id}.txt`

## Environment Variables

```
CLOUDFLARE_API_TOKEN=
DATABASE_ID=
ACCOUNT_ID=
```

## Ideal Use Cases

- Healthcare lecture transcription  
- Doctor–patient conversation logs  
- YouTube video transcription  
- Large-audio chunk processing  
- Multilingual → English translation  
