import google.generativeai as genai
import os
import time
import json
import subprocess
import requests
import base64
import numpy as np
import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
import pymysql
import subprocess
from pydub import AudioSegment
from google.cloud import storage

from datetime import datetime,timedelta
import google.generativeai as genai
from datetime import datetime, timedelta

def download_mp4(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def convert_mp4_to_mp3(mp4_path, mp3_path):
    command = [
        'ffmpeg',
        '-i', mp4_path,
        '-q:a', '0',
        '-map', 'a',
        mp3_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg Error Output:")
        print(e.stderr)
        raise

def chunk_audio(input_file, chunk_duration_minutes=5):
    """Split audio file into chunks using FFmpeg (handles files of any size)"""
    try:
        probe_command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_file
        ]
        
        print(f"Getting duration of {input_file}...")
        result = subprocess.run(probe_command, capture_output=True, text=True, check=True)
        total_duration = float(result.stdout.strip())
        print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        
        chunk_duration_seconds = chunk_duration_minutes * 60
        chunks = []
        
        num_chunks = int(total_duration / chunk_duration_seconds) + 1
        print(f"Will create approximately {num_chunks} chunks")
        
        chunk_index = 0
        for start_time in range(0, int(total_duration), chunk_duration_seconds):
            chunk_path = f"chunk_{chunk_index}.mp3"
            
            command = [
                'ffmpeg',
                '-ss', str(start_time),
                '-t', str(chunk_duration_seconds),
                '-i', input_file,
                '-acodec', 'libmp3lame',
                '-q:a', '2',
                '-y',
                chunk_path
            ]
            
            print(f"Creating chunk {chunk_index}: {chunk_path} (starting at {start_time}s / {start_time/60:.1f}m)")
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Warning: FFmpeg returned error for chunk {chunk_index}")
                print(f"Error: {result.stderr}")
                continue
            
            if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 1000:
                chunks.append((chunk_path, start_time * 1000))
                print(f"✓ Chunk {chunk_index} created successfully ({os.path.getsize(chunk_path) / 1024 / 1024:.2f} MB)")
            else:
                print(f"✗ Chunk {chunk_index} failed or too small")
            
            chunk_index += 1
        
        print(f"\n✓ Successfully created {len(chunks)} chunks")
        return chunks
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if hasattr(e.stderr, 'decode') else str(e.stderr)
        print(f"FFmpeg/FFprobe error: {error_msg}")
        raise
    except Exception as e:
        print(f"Error in chunk_audio: {str(e)}")
        raise

def format_timestamp(milliseconds):
    """Convert milliseconds to HH:MM:SS format"""
    total_seconds = int(milliseconds / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_chunk(chunk_info, api_key):
    """Process a single audio chunk through Gemini using base64 inline data"""
    chunk_path, time_offset = chunk_info
    
    try:
        print(f"\nProcessing chunk: {chunk_path}")
        
        # Re-configure genai for each chunk to ensure clean state
        genai.configure(api_key=api_key)
        
        # Read audio file as base64 (workaround for ragStoreName issue)
        print(f"Reading {chunk_path} as base64...")
        with open(chunk_path, 'rb') as f:
            audio_bytes = f.read()
            audio_data = base64.standard_b64encode(audio_bytes).decode('utf-8')
        
        file_size_mb = len(audio_bytes) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        prompt = """Instructions:
                    CRITICAL: Language Requirements
                    - ALWAYS provide the transcript in English only
                    - If there is Hindi or any other language speech, translate it to English
                    - Do NOT include any non-English text in the output
                    - Ensure the translation maintains the original meaning while being natural in English
                    
                    Domain-Specific Focus: 
                    - This transcript is in the healthcare domain
                    - Ensure medical terminology, drug names, and healthcare-related phrases are accurately transcribed and translated to English
                    
                    Segment Length: 
                    - Break the transcript into natural, easily readable segments
                    - Each segment should be 20-25 seconds long
                    - Maintain context and readability in the English translation
                    
                    Output Format: 
                    - Provide output as JSON objects with 'time' and 'text' fields
                    - Timestamps should be relative to chunk start (00:00:00)
                    - Text must ALWAYS be in English
                    
                    Example Format:
                    { "time": "00:00:57", "text": "The patient reported mild chest pain lasting for about 10 minutes." },
                    { "time": "00:01:15", "text": "No history of hypertension or diabetes was noted in the medical records." }
                    
                    Critical Guidelines:
                    - Convert ALL speech to English - no exceptions
                    - Translate any Hindi or other language content to natural-sounding English
                    - Keep segments meaningful and contextual (20-25 seconds)
                    - Use precise timestamps relative to chunk start
                    - Include only speech content
                    - Ensure valid JSON format
                    - Double-check that NO Hindi or non-English text remains in output
        """
        
        # Create a fresh model instance for each chunk
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 20950,
            "response_mime_type": "text/plain",
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config,
        )
        
        # Generate content with inline audio data (no file upload needed)
        print(f"Sending audio data to Gemini API...")
        response = model.generate_content([
            {
                'mime_type': 'audio/mpeg',
                'data': audio_data
            },
            prompt
        ])
        
        print(f"\nRaw response for {chunk_path}:")
        print(response.text)
        
        # Extract JSON objects from response
        text = response.text.strip()
        json_objects = []
        
        start = 0
        while True:
            start = text.find('{', start)
            if start == -1:
                break
            count = 1
            end = start + 1
            
            while count > 0 and end < len(text):
                if text[end] == '{':
                    count += 1
                elif text[end] == '}':
                    count -= 1
                end += 1
            
            if count == 0:
                try:
                    obj = json.loads(text[start:end])
                    if 'time' in obj and 'text' in obj:
                        time_parts = obj['time'].split(':')
                        if len(time_parts) == 3:
                            timestamp_ms = (int(time_parts[0]) * 3600 + 
                                         int(time_parts[1]) * 60 + 
                                         int(time_parts[2])) * 1000
                            timestamp_ms += time_offset
                            obj['time'] = format_timestamp(timestamp_ms)
                            obj['text'] = obj['text'].strip()
                            json_objects.append(obj)
                except json.JSONDecodeError:
                    pass
                
            start = end
        
        if not json_objects:
            raise ValueError("No valid transcript segments found in response")
            
        return json_objects
            
    except Exception as e:
        print(f"\nError processing {chunk_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return [{"time": format_timestamp(time_offset), 
                "text": f"[Error processing chunk {os.path.basename(chunk_path)}: {str(e)}]"}]

def merge_transcripts(results):
    """Merge transcript chunks and sort by timestamp"""
    all_segments = []
    for chunk_result in results:
        all_segments.extend(chunk_result)
    
    all_segments.sort(key=lambda x: x['time'])
    
    filtered_segments = []
    prev_time = None
    for segment in all_segments:
        curr_time = segment['time']
        if prev_time != curr_time:
            filtered_segments.append(segment)
        prev_time = curr_time
            
    return filtered_segments

def process_audio(mp3_path, output_filename):
    """Main processing function"""
    CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
    DATABASE_ID = os.getenv("DATABASE_ID")
    ACCOUNT_ID = os.getenv("ACCOUNT_ID")
    
    API_ENDPOINT = f'https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/d1/database/{DATABASE_ID}/query'
    
    headers = {
        'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    query = {
        "sql": "select * from gemini_token order by timestamp asc limit 1"
    }
    
    response = requests.post(API_ENDPOINT, headers=headers, json=query)
    a = response.json()
    g_t = a['result'][0]['results'][0]['token']
    
    current_datetime = datetime.today()
    adjusted_datetime = current_datetime + timedelta(minutes=330)
    formatted_datetime = adjusted_datetime.strftime("%Y-%m-%d %H:%M:%S")
    
    insert_query = {
        "sql": f"UPDATE gemini_token SET timestamp = '{formatted_datetime}' WHERE token = '{g_t}'"
    }
    
    response = requests.post(API_ENDPOINT, headers=headers, json=insert_query)
    
    if response.status_code == 200:
        data = response.json()
        print("Row updated successfully:", data)
    else:
        print(f'Error: {response.status_code}')
        print(response.json())
    
    # Process audio in chunks
    chunks = chunk_audio(mp3_path)
    results = []
    
    for chunk_info in chunks:
        try:
            # Pass API key instead of model object
            result = process_chunk(chunk_info, g_t)
            if result:
                results.append(result)
        finally:
            if os.path.exists(chunk_info[0]):
                os.remove(chunk_info[0])
    
    # Merge and save results
    final_result = merge_transcripts(results)
    
    output_path = f"{output_filename}.json"
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete. Results saved to {output_path}")
    return final_result
    
mp4_url = z['original_url']
xy = z['service_id']
print("xy",xy)
mp4_path = f"{xy}.mp4"
mp3_path = f"{xy}.mp3"


download_mp4(mp4_url, mp4_path)

convert_mp4_to_mp3(mp4_path, mp3_path)

mp3_path= mp3_path
output_filename=xy
final = process_audio(mp3_path, output_filename)

print("\nFinal result:")
json_str = json.dumps(final, indent=2, ensure_ascii=False)

storage_client = storage.Client.from_service_account_json("service_account.json")
bucket_name = 'video_transcript'
bucket = storage_client.bucket(bucket_name)
blob_name = f"gemini_transcript/{output_filename}.txt"
blob = bucket.blob(blob_name)
import json


json_file = f'''{output_filename}.json'''  # Replace with your JSON file path
txt_file = f"{output_filename}.txt"  # Path for the output .txt file

try:
    with open(json_file, "r") as f:
        data = json.load(f)  # Load JSON content

    # Write the content into a .txt file
    with open(txt_file, "w") as f:
        # If you want to pretty print the JSON:
        f.write(json.dumps(data))
        
    print(f"Successfully created {txt_file} from {json_file}!")

except Exception as e:
    print(f"An error occurred: {e}")

blob.upload_from_filename(txt_file)
os.remove(mp3_path)
os.remove(json_file)
os.remove(mp4_path)
os.remove(txt_file)
lst=[]
x=f'''https://storage.googleapis.com/video_transcript/gemini_transcript/{xy}.txt'''
d={}
d['content_id']=xy
d['srt']=x
lst.append(d)
df=pd.DataFrame(lst)

lst=[]
for i in df.to_dict("records"):
    url = i['srt']

    response = requests.get(url)

    if response.status_code == 200:
        content = response.text
    i.update({"text":content})
    lst.append(i)
df=pd.DataFrame(lst)
df['type']=2
df.drop("srt",axis=1,inplace=True)
df.rename({"content_id":"type_id","text":"srt"},axis=1,inplace=True)
df=df[['type','type_id','srt']]
