# Slack Puzzle Bot

A Slack bot that processes puzzle results shared in a channel, calculates scores, and maintains a leaderboard.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contribute](#contribute)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- pip (Python's package installer)
- A Slack workspace and the necessary credentials (bot token, signing secret)

### Steps

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/chrisaberle/connections_bot
    cd connections_bot
    ```

2. **Setup Virtual Environment:**

    ```bash
    poetry install
    ```
   
    Poetry will automatically create a virtual environment and install dependencies when installing the project. To activate the created virtual environment, you can use:
    
    ```bash
   poetry shell
   ```

3. **Setup Environment Variables:**

   Create a `.env` file in the root directory and define the following variables:
   
    ```env
    SLACK_BOT_TOKEN=[Your Slack Bot Token]
    SLACK_SIGNING_SECRET=[Your Slack Signing Secret]
    ```

    - `SLACK_BOT_TOKEN`: Token for the bot user installed in your Slack workspace.
    - `SLACK_SIGNING_SECRET`: Secret used to verify requests from Slack.


    **Note:** Keep your `.env` file private and never expose your secrets.


4. **Setup Database:**

    Ensure that all necessary CSV files (`historic_scores.csv`, `users.csv`) are set up with the appropriate columns.

## Usage

### Running the Application

1. **Start the Server:**

    ```bash
    uvicorn main:app --reload
    ```

2. **Interact with the Bot:**

    With the application running, interact with the bot on Slack by sharing puzzle results and using the `/update` and `/leaderboard` commands.