from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
import pandas as pd
import time
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, Form
import hashlib
import hmac
import json
import os
from pydantic import BaseModel
from typing import Optional, Dict
import logging
import aiohttp
import re
import asyncio


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Slack signing secret
SLACK_SIGNING_SECRET = os.environ['SLACK_SIGNING_SECRET'].encode('utf-8')

# FastAPI instance
app = FastAPI()

# Define data models
class SlackEvent(BaseModel):
    token: str
    type: str
    challenge: Optional[str] = None
    event: Optional[Dict] = None


class SlashCommand(BaseModel):
    token: str
    team_id: str
    channel_id: str
    user_id: str
    command: str
    text: str
    response_url: str

"""
FastAPI Routes
"""


@app.post("/slack/commands/update")
async def update_command(request: Request, background_tasks: BackgroundTasks):
    # [ ... ] (No changes in the initial part of the function)

    # Parse command
    command_data = await request.form()

    # Fetch all messages
    messages = await fetch_all_messages(command_data['channel_id'])

    # Move the processing and updating tasks to background tasks
    background_tasks.add_task(process_all_messages, messages)
    background_tasks.add_task(update_users)
    background_tasks.add_task(prune_historic_data)

    # Send a response back to Slack immediately
    return {"response_type": "in_channel", "text": "Data refresh initiated! Your leaderboard is now as brisk as March in Petaluma."}


@app.post("/slack/commands/leaderboard")
async def leaderboard_command(request: Request):
    # Get Slack request headers
    slack_signature = request.headers.get("X-Slack-Signature")
    slack_request_timestamp = request.headers.get("X-Slack-Request-Timestamp")

    # Get request body
    body = await request.body()
    body_str = body.decode()

    # Validate request
    req = str.encode(f"v0:{slack_request_timestamp}:") + body
    request_hash = 'v0=' + hmac.new(SLACK_SIGNING_SECRET, req, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(request_hash, slack_signature):
        raise HTTPException(status_code=400, detail="Invalid request")

    # Parse command
    command = await request.form()

    # Generate leaderboard message
    leaderboard_message = await generate_leaderboard_message()

    # Send message back to Slack
    return {"response_type": "in_channel", "text": leaderboard_message}


@app.post("/slack/events")
async def handle_slack_event(request: Request, background_tasks: BackgroundTasks):
    # Get Slack request headers
    slack_signature = request.headers.get("X-Slack-Signature")
    slack_request_timestamp = request.headers.get("X-Slack-Request-Timestamp")

    # Get request body
    body = await request.body()
    body_str = body.decode()

    # Validate request
    req = str.encode(f"v0:{slack_request_timestamp}:") + body
    request_hash = 'v0=' + hmac.new(SLACK_SIGNING_SECRET, req, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(request_hash, slack_signature):
        raise HTTPException(status_code=400, detail="Invalid request")

    # Parse Slack event data
    event_data = json.loads(body_str)

    # URL Verification Challenge
    if "challenge" in event_data:
        return {"challenge": event_data["challenge"]}

    # Use background task for processing and immediately acknowledge
    background_tasks.add_task(process_event, event_data)

    return {"status": "ok"}


"""
Utility Functions
"""

def prune_historic_data():
    # Load and sort the historic data
    historic_data = pd.read_csv('historic_scores.csv')
    historic_data = historic_data.sort_values('timestamp')

    # Drop duplicate entries based on 'user' and 'puzzle_number', keeping the first (or last) occurrence
    historic_data_pruned = historic_data.drop_duplicates(subset=['user', 'puzzle_number'], keep='first')

    # Save the new data set
    historic_data_pruned.to_csv('historic_scores.csv', index=False)

async def fetch_all_messages(channel_id):
    base_url = "https://slack.com/api/conversations.history"
    slack_token = os.environ['SLACK_BOT_TOKEN']

    all_messages = []
    cursor = None

    headers = {
        "Authorization": f"Bearer {slack_token}"
    }

    async with aiohttp.ClientSession() as session:
        try:  # Add a try-except block to catch and log exceptions
            while True:
                params = {
                    "channel": channel_id,
                    "limit": 100
                }
                if cursor:
                    params["cursor"] = cursor

                logger.info("Fetching messages...")  # Log before the API call
                async with session.get(base_url, params=params, headers=headers) as response:
                    data = await response.json()
                    logger.info(f"Fetched {len(data['messages'])} messages.")  # Log the number of messages fetched

                    # Check for API error
                    if not data['ok']:
                        raise ValueError("Slack API request failed with error: {}".format(data['error']))

                    all_messages.extend(data['messages'])
                    cursor = data.get('response_metadata', {}).get('next_cursor')
                    logger.info(f"Cursor: {cursor}")  # Log the cursor

                    if not cursor:
                        break

                await asyncio.sleep(1)  # Respect rate limits
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")  # Log exceptions

    logger.info(f"Total messages fetched: {len(all_messages)}")
    return all_messages


async def process_all_messages(messages):
    processed_data = []

    # Load historic scores
    historic_scores_df = pd.read_csv('historic_scores.csv')

    # Ensure 'puzzle_number' is of type int in the historic data
    historic_scores_df['puzzle_number'] = historic_scores_df['puzzle_number'].astype(int)

    # Iterate through messages
    for idx, message in enumerate(messages):
        # Check if the message is a valid puzzle message
        if is_api_puzzle_message(message):
            # Extract and calculate necessary information
            user = message['user']
            text = message['text']
            timestamp = message['ts']
            score = calculate_score(text)
            puzzle_number = int(extract_puzzle_number(text))  # Ensure it's an integer

            # Append to processed_data without checking for duplicates
            processed_data.append({
                'user': user,
                'score': score,
                'timestamp': timestamp,
                'puzzle_number': puzzle_number
            })
            logger.info(f"Message {idx + 1}: Processed data: {processed_data[-1]}")
        else:
            logger.info(f"Message {idx + 1}: Not a puzzle message, skipped.")

    # Convert to DataFrame and return
    processed_df = pd.DataFrame(processed_data)
    all_data_df = pd.concat([historic_scores_df, processed_df], ignore_index=True)
    all_data_df.to_csv('historic_scores.csv', index=False)

    return processed_df


async def update_users():
    try:
        # Define the Slack client within the function
        slack_token = os.environ["SLACK_BOT_TOKEN"]
        client = WebClient(token=slack_token)

        # Fetch all users from Slack API
        api_response = client.users_list()
        users = api_response['members']

        # Extract user id and name
        user_data = [{
            'user': user['id'],
            'username': user['profile']['real_name']
        } for user in users]

        # Save to users.csv
        users_df = pd.DataFrame(user_data)
        users_df.to_csv('users.csv', index=False)

    except SlackApiError as e:
        logger.info(f"Error fetching users: {e.response['error']}")


async def process_event(event):
    try:
        # Log the entire event payload
        logger.info(f"Received event: {event}")

        # Check if the event is a message and contains a puzzle result
        if is_puzzle_message(event):
            logger.info("Identified as a puzzle message")

            # Process puzzle message: add to CSVs, etc.
            result = await process_puzzle_message(event)

            logger.info(f"Processing result: {result}")

            # If successful, calculate and send score
            if result == "success":
                username = await get_username(event['event']['user'])
                score = calculate_score(event['event']['text'])
                logger.info(f"Calculated score: {score}")

                await send_slack_message(event['event']['channel'], f"Good job, {username}! Your score is {score}.")
            elif result == "invalid":
                await send_slack_message(event['event']['channel'],
                                         "Invalid puzzle share. Report to Jim for his quarterly Ooni cleaning immediately.")
            elif result == "duplicate":
                puzzle_number = extract_puzzle_number(event['event']['text'])
                await send_slack_message(event['event']['channel'],
                                         f"You've already submitted a score for Puzzle #{puzzle_number}. Duplicate entries are not allowed. Darius and Shirley the Ostrich await you in the Battle Dome. Good luck.")
            else:
                logger.warning(f"Unexpected result value: {result}")
        else:
            logger.info("Not identified as a puzzle message.")

    except Exception as e:
        logger.error(f"Error handling request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def is_api_puzzle_message(message) -> bool:
    return (
        message['type'] == 'message' and
        'text' in message and
        'Connections\nPuzzle #' in message['text']
    )


async def generate_leaderboard_message():
    # Load historic scores
    historic_scores_df = pd.read_csv('historic_scores.csv')

    # Load user names
    users = pd.read_csv('users.csv')

    # Calculate total and average scores
    summary_scores = historic_scores_df.groupby('user')['score'].agg(['sum', 'mean']).reset_index()
    summary_scores.columns = ['user', 'total_score', 'average_score']

    # Merge with usernames and sort by total_score
    leaderboard_df = pd.merge(summary_scores, users, on='user')
    leaderboard_df = leaderboard_df.sort_values(by='total_score', ascending=False).head(5)

    # Format message
    title_line = ":trophy: *LEADERBOARD* :trophy:\n\n"

    message_lines = [
        f"{idx + 1} | {row.username} | {row.total_score} total points | {row.average_score:.2f} average"
        for idx, row in enumerate(leaderboard_df.itertuples())
    ]

    # Combine all message components
    return "\n".join([title_line] + message_lines)


async def get_username(user_id: str) -> str:
    try:
        slack_token_async = os.environ['SLACK_BOT_TOKEN']
        client_async = AsyncWebClient(token=slack_token_async)

        user_info_response = await client_async.users_info(user=user_id)
        if user_info_response['ok']:
            return user_info_response['user']['name']
        else:
            logging.error(f"Couldn't fetch username for {user_id}: {user_info_response['error']}")
            return None
    except Exception as e:
        logging.error(f"Error fetching username for {user_id}: {str(e)}")
        return None


async def send_slack_message(channel: str, message: str):
    url = "https://slack.com/api/chat.postMessage"
    payload = {
        "channel": channel,
        "text": message,
    }
    headers = {
        "Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        await session.post(url, data=json.dumps(payload), headers=headers)


def is_puzzle_message(event) -> bool:
    return (
        event['event']['type'] == 'message' and
        'text' in event['event'] and
        'Connections\nPuzzle #' in event['event']['text']
    )


def is_valid_score_message(text, square_dict):
    # Replace square names in text with single-character emojis
    for square_name, square_char in square_dict.items():
        text = text.replace(square_name, square_char)

    # Identify and count the squares
    squares = "🟪🟦🟨🟩"
    square_count = sum(1 for char in text if char in squares)

    # Check the conditions
    return square_count >= 16 and square_count % 4 == 0


def extract_puzzle_number(puzzle_result):
    # Extract and return the puzzle number from the puzzle result string
    match = re.search(r"Puzzle #(\d+)", puzzle_result)
    return int(match.group(1)) if match else None


async def process_puzzle_message(event):
    message_text = event['event']['text']

    # Check for valid message
    square_dict = {
        ":large_purple_square:": "🟪",
        ":large_blue_square:": "🟦",
        ":large_yellow_square:": "🟨",
        ":large_green_square:": "🟩"
    }
    if not is_valid_score_message(message_text, square_dict):
        # Log a message or notify the user in the channel
        logger.warning(f"Invalid puzzle message from user {event['event']['user']}: {message_text}")

        return "invalid"

    # Add to historic_scores.csv
    historic_scores_df = pd.read_csv('historic_scores.csv')

    # Extract puzzle number
    puzzle_number = extract_puzzle_number(message_text)

    # Check for duplicate entry
    user = event['event']['user']
    is_duplicate = (
            (historic_scores_df['user'] == user) &
            (historic_scores_df['puzzle_number'] == puzzle_number)
    ).any()

    if is_duplicate:
        logger.warning(f"Duplicate entry from user {user} for puzzle {puzzle_number}")

        return "duplicate"

    # Calculate score
    score = calculate_score(message_text)

    new_row = pd.DataFrame({
        'user': [event['event']['user']],
        'score': [score],
        'timestamp': [event['event']['ts']],
        'puzzle_number': [extract_puzzle_number(message_text)]
    })
    historic_scores_df = pd.concat([historic_scores_df, new_row], ignore_index=True)
    historic_scores_df.to_csv('historic_scores.csv', index=False)

    return "success"


def calculate_score(text):
    square_dict = {
        ":large_purple_square:": "🟪",
        ":large_blue_square:": "🟦",
        ":large_yellow_square:": "🟨",
        ":large_green_square:": "🟩"
    }
    score = 0

    # Replace square names in text with single-character emojis
    for square_name, square_char in square_dict.items():
        text = text.replace(square_name, square_char)

    # Filter squares from the text and chunk them into attempts of size 4
    squares = "🟪🟦🟨🟩"
    flat = [t for t in text if t in squares]
    attempts = [flat[i:i + 4] for i in range(0, len(flat), 4)]

    # Scoring logic
    for round, guesses in enumerate(attempts):
        inc = 0
        if all(g == "🟪" for g in guesses):
            inc += 10
        elif all(g == "🟦" for g in guesses):
            inc += 8
        elif all(g == "🟨" for g in guesses):
            inc += 6
        elif all(g == "🟩" for g in guesses):
            inc += 4
        if round <= 4: inc *= (4 - round)
        score += inc

    return score


def get_user_list():
    slack_token = os.environ['SLACK_BOT_TOKEN']
    client = WebClient(token=slack_token)

    response = client.users_list()
    users = response['members']

    user_data = {
        'user': [user['id'] for user in users],
        'username': [user['name'] for user in users]
    }

    users_df = pd.DataFrame(user_data)
    users_df.to_csv('users.csv', index=False)

    return users_df


# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)