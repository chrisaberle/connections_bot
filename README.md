# Slack Puzzle Bot 🧩

A Slack bot designed to process NYT Connections puzzle results shared within a Slack channel. It calculates scores, maintains a leaderboard, and provides statistics on-demand.

## Table of Contents

- [Features](#features)
- [Local Installation](#local-installation)
- [Setting Up a Slack Bot](#setting-up-a-slack-bot)
- [Usage](#usage)
- [Contribute](#contribute)
- [License](#license)

## Features

- **Update Scores**: Automatically processes puzzle results and updates scores.
- **Leaderboard**: Displays a leaderboard with rankings based on scores.
- **Statistics**: Provides individual and overall puzzle statistics.

## Local Installation

### Prerequisites

- Docker and Docker Compose.
- [ngrok](https://ngrok.com/download) - For creating a secure tunnel to your localhost.
- Python 3.8+ (if running outside Docker).
- A Slack workspace with necessary credentials (bot token, signing secret).

### Steps

1. **Clone the Repository**:
   
   ```bash
   git clone https://github.com/chrisaberle/connections_bot
   cd connections_bot
   ````

2. **Setup Environment Variables**
   
   Create a `.env` file in the root directory with the following variables. Keep your `.env` file private; never expose your secrets.
   ```
   SLACK_BOT_TOKEN=[Your Slack Bot Token]
   SLACK_SIGNING_SECRET=[Your Slack Signing Secret]
   ```

3. **Run Using Docker Compose**

   ```bash
   docker-compose up --build
   ```
   
4. **Setup ngrok**

   ```bash
   ngrok http 8000
   ```
   
   Note down the HTTPS URL provided by ngrok. This will be used for your Slack app's event and command subscriptions.

## Setting Up a Slack Bot

1. **Create a New Slack App**:
   
   - Visit [Slack API's "Your Apps" page](https://api.slack.com/apps) and create a new app.

2. **Permissions**:
   
   - Add the `chat:write` permission under OAuth & Permissions to allow the bot to post messages.
   - Install the bot to your workspace.

3. **Commands**:

   Under "Slash Commands", create the following commands:

   - `/update` - Processes the channel's puzzle results.
   - `/leaderboard` - Displays the current leaderboard.
   - `/stats` - Provides individual puzzle statistics.

   For each command, set the Request URL to the HTTPS URL from ngrok followed by the endpoint path.

4. **Event Subscriptions and OAuth Scopes**:

   Enable event subscriptions and set the Request URL. Add OAuth scopes relevant to your installation, keeping in mind the principle of least privilege. 
`message.channels`, `channels:read` and `chat:write` are some good examples.

5. **Install the App to Workspace**:

   Reinstall the app to your workspace to reflect the changes.

## Usage

1. **Perform Initial Data Pull**:

   After setting up the bot on Slack, run an initial `/update` command. This processes the entire channel history, creating and populating the .csv databases.

2. **Interact with the Bot**:

   Share puzzle results in your channel. As you post scores, the bot will track and update the data. Use `/leaderboard` to view rankings and `/stats` for statistics. Remember to use `/update` sparingly to avoid rate limits.

## Contribute

This is a private project and I have no idea how to use Github. Contributions are welcome but be ready to teach me about pull requests