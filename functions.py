import requests
import json
import polars as pl
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import os

def getVideoRecords(response):
    try:
        # Attempt to parse the response as JSON
        response_data = json.loads(response.text)

        # Check if 'items' exists in the parsed data
        if 'items' in response_data:
            return response_data['items']
        else:
            print("No 'items' key found in the response")
            return []  # Return an empty list if 'items' is not found

    except json.JSONDecodeError:
        print("Failed to decode JSON response. Response text:")
        print(response.text)  # Print the raw response for debugging
        return []  # Return an empty list on decode failure

    except KeyError as e:
        print(f"Error: {e}. The response does not contain the expected structure.")
        return []  # Return an empty list on KeyError

def getVideoIDs():
    """
        Function to return all video IDs for Shaw Talebi's YouTube channel

        Dependencies: 
            - getVideoRecords()
    """
    channel_id = 'UC0y9s4PwBwMYciq73UTe3XA'  # My YouTube channel ID
    page_token = None  # Initialize page token
    url = 'https://www.googleapis.com/youtube/v3/search'  # YouTube search API endpoint
    my_key = os.getenv('YT_API_KEY')

    # Extract video data across multiple search result pages
    video_record_list = []

    while page_token != 0:
        params = {
            "key": my_key,
            'channelId': channel_id,
            'part': ["snippet", "id"],
            'order': "date",
            'maxResults': 50,
            'pageToken': page_token
        }
        response = requests.get(url, params=params)

        # Append video records to list
        video_record_list += getVideoRecords(response)

        try:
            # Grab next page token
            page_token = json.loads(response.text)['nextPageToken']
        except:
            # If no next page token, kill while loop
            page_token = 0

    # Write video IDs as CSV file
    video_ids_df = pl.DataFrame(video_record_list)
    video_ids_df.write_csv('data/video-ids.csv')

def extractTranscriptText(transcript: list) -> str:
    """
        Function to extract text from transcript dictionary

        Dependencies:
            - getVideoTranscripts()
    """
    text_list = [transcript[i]['text'] for i in range(len(transcript))]
    return ' '.join(text_list)

def getVideoTranscripts():
    """
        Function to extract transcripts for all video IDs stored in "data/video-ids.csv"

        Dependencies:
            - extractTranscriptText()
    """
    df = pl.read_csv('data/video-ids.csv')

    transcript_text_list = []

    for i in range(len(df)):
        # Try to extract captions
        try:
            transcript = YouTubeTranscriptApi.get_transcript(df['video_id'][i])
            transcript_text = extractTranscriptText(transcript)
        # If not available, set as n/a
        except:
            transcript_text = "n/a"
        
        transcript_text_list.append(transcript_text)

    # Add transcripts to dataframe
    df = df.with_columns(pl.Series(name="transcript", values=transcript_text_list))

    # Write dataframe to CSV file
    df.write_csv('data/video-transcripts.csv')

def handleSpecialStrings(df: pl.DataFrame) -> pl.DataFrame:
    """
        Function to replace special character strings in video transcripts and titles
        
        Dependencies:
            - transformData()
    """
    special_strings = ['&#39;', '&amp;', 'sha ']
    special_string_replacements = ["'", "&", "Shaw "]

    for i in range(len(special_strings)):
        df = df.with_columns(
            df['title'].str.replace(special_strings[i], special_string_replacements[i]).alias('title')
        )
        df = df.with_columns(
            df['transcript'].str.replace(special_strings[i], special_string_replacements[i]).alias('transcript')
        )

    return df

def setDatatypes(df: pl.DataFrame) -> pl.DataFrame:
    """
        Function to change data types of columns in polars data frame containing video IDs, dates, titles, and transcripts

        Dependencies:
            - transformData()
    """
    # Change datetime to Datetime dtype
    df = df.with_columns(pl.col('datetime').cast(pl.Datetime))

    return df

def transformData():
    """
        Function to preprocess video data

        Dependencies:
            - handleSpecialStrings()
            - setDatatypes()
    """
    df = pl.read_csv('data/video-transcripts.csv')

    df = handleSpecialStrings(df)
    df = setDatatypes(df)

    # Write the processed data back to CSV
    df.write_csv('data/video-transcripts.csv')

def createTextEmbeddings():
    """
        Function to generate text embeddings of video titles and transcripts
    """
    # Read data from file
    df = pl.read_csv('data/video-transcripts.csv')

    # Define embedding model and columns to embed
    model = SentenceTransformer('all-MiniLM-L6-v2')

    column_name_list = ['title', 'transcript']

    for column_name in column_name_list:
        # Generate embeddings
        embedding_arr = model.encode(df[column_name].to_list())

        # Store embeddings in a dataframe
        schema_dict = {f"{column_name}_embedding-{i}": float for i in range(embedding_arr.shape[1])}
        df_embedding = pl.DataFrame(embedding_arr, schema=schema_dict)

        # Append embeddings to video index
        df = pl.concat([df, df_embedding], how='horizontal')

    # Write data to CSV
    df.write_csv('data/video-index.csv')
