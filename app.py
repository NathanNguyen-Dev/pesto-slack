import os
import logging # Added for logging
from openai import OpenAI # Updated import
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

# Import Supabase client
from supabase_client import supabase

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# --- In-memory store for active opt-in sessions ---
active_opt_ins = {} # Stores user_id: {"current_question_index": 0, "answers": {}, "thread_ts": None}

# --- Define Opt-in Questions ---
OPT_IN_QUESTIONS = [
    {"id": "full_name", "prompt": "Let's start with your Pesto Profile! What's your full name?", "column": "full_name"},
    {"id": "location", "prompt": "Which city and country are you based in?", "column": "location"},
    {"id": "linkedin_url", "prompt": "Please share your LinkedIn profile URL (e.g., https://linkedin.com/in/yourprofile).", "column": "linkedin_url"},
    {"id": "company", "prompt": "Which company or organization are you with?", "column": "company"},
    {"id": "industry", "prompt": "What's your primary industry or field?", "column": "industry"},
    {"id": "skillset", "prompt": "Please list 3–5 core skillsets you bring (e.g., React, data analysis, product management).", "column": "skillset"},
    {"id": "background_summary", "prompt": "Briefly describe your professional background or career highlights.", "column": "background_summary"},
    {"id": "current_projects", "prompt": "What are you currently working on or building?", "column": "current_projects"},
    {"id": "topics_of_interest", "prompt": "Please list 3–5 topics you're most interested in (e.g., AI, blockchain, UX design, growth marketing).", "column": "topics_of_interest"},
    {"id": "networking_goals", "prompt": "And finally, what is your top networking goal? (e.g., finding a mentor, an expert, a co-founder)", "column": "networking_goals"}
]

# --- Embedding Helper Function (Updated for openai >= 1.0.0) ---
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    """Generates an embedding for the given text using OpenAI."""
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=[text], # API expects a list of texts
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        # Use the main logger, not logger from __main__ if this is a helper module
        # For app.py, using logger (defined globally) is fine.
        logger.error(f"Error generating embedding: {e}", exc_info=True)
        return None

# --- Helper function to ask a question ---
def ask_question(user_id, client):
    session = active_opt_ins.get(user_id)
    if not session:
        return

    question_index = session["current_question_index"]
    thread_ts_to_use = session.get("thread_ts") # Get thread_ts from session

    if question_index < len(OPT_IN_QUESTIONS):
        question_data = OPT_IN_QUESTIONS[question_index]
        try:
            payload = {
                "channel": user_id, 
                "text": question_data["prompt"]
            }
            if thread_ts_to_use:
                payload["thread_ts"] = thread_ts_to_use
            
            client.chat_postMessage(**payload)
            logger.info(f"Asked question {question_index} to user {user_id} (thread_ts: {thread_ts_to_use})")
        except Exception as e:
            logger.error(f"Error sending question to user {user_id}: {e}")
    else:
        # All questions answered, this part is mostly handled in handle_message_events now
        logger.info(f"All questions answered by user {user_id}. Finalizing (called from ask_question, should be rare).")
        try:
            payload = {
                "channel": user_id, 
                "text": "Thanks! I have all your answers. Processing your profile..."
            }
            if thread_ts_to_use:
                payload["thread_ts"] = thread_ts_to_use
            client.chat_postMessage(**payload)
        except Exception as e:
            logger.error(f"Error sending completion message from ask_question to user {user_id}: {e}")

# --- /opt-in Command Handler (Conversational) ---
@app.command("/opt-in")
def handle_opt_in_command_conversational(ack, body, client, logger):
    """Handles the /opt-in slash command and starts the conversational opt-in flow."""
    ack()
    user_id = body["user_id"]

    if user_id in active_opt_ins and active_opt_ins[user_id]["current_question_index"] < len(OPT_IN_QUESTIONS):
        try:
            current_session_thread_ts = active_opt_ins[user_id].get("thread_ts")
            client.chat_postMessage(
                channel=user_id,
                text="It looks like you're already in the process of opting in. Let's continue where you left off.",
                thread_ts=current_session_thread_ts # Continue in existing thread if applicable
            )
            ask_question(user_id, client) # Re-ask current question, respecting thread_ts via session
        except Exception as e:
            logger.error(f"Error reminding user {user_id} about ongoing opt-in: {e}")
    else:
        # Start a new session, ensure thread_ts is initialized to None
        active_opt_ins[user_id] = {
            "current_question_index": 0, 
            "answers": {},
            "thread_ts": None # Initialize thread_ts for the new session
        }
        try:
            client.chat_postMessage(
                channel=body["channel_id"],
                text=f"<@{user_id}>, I'll send you a DM to start setting up your Pesto profile! Please check your DMs."
            )
            client.chat_postMessage(
                channel=user_id,
                text="Hi there! Thanks for your interest in Pesto. I'll ask you a few questions to set up your profile. You can reply to me directly here."
                # First message in DM is not threaded
            )
            ask_question(user_id, client) # Ask the first question (will not be threaded yet)
        except Exception as e:
            logger.error(f"Error starting opt-in conversation with user {user_id}: {e}")

# --- Message Event Handler ---
@app.event("message")
def handle_message_events(event, client, logger):
    logger.info(f"--- Message Event Received ---")
    logger.info(f"Event details: {event}")

    if event.get("bot_id"):
        logger.info("Message from a bot, ignoring.")
        return

    channel_type = event.get("channel_type")
    logger.info(f"Message channel_type: {channel_type}")
    if channel_type != "im":
        logger.info("Message not in a DM (channel_type is not 'im'), ignoring for opt-in flow.")
        return

    user_id = event.get("user")
    logger.info(f"Message from user_id: {user_id}")
    if user_id not in active_opt_ins:
        logger.warning(f"User {user_id} sent a DM, but is not in an active opt-in session. Current sessions: {list(active_opt_ins.keys())}")
        return

    logger.info(f"User {user_id} is in an active opt-in session.")
    session = active_opt_ins[user_id]
    question_index = session["current_question_index"]
    logger.info(f"Current question_index for user {user_id}: {question_index}")

    # Update session's thread_ts based on the incoming message's thread context
    session["thread_ts"] = event.get("thread_ts") 
    logger.info(f"Updated session thread_ts for user {user_id} to: {session['thread_ts']}")
    
    thread_ts_to_use = session.get("thread_ts") # Use this for replies in this handler

    if question_index < len(OPT_IN_QUESTIONS):
        answer = event.get("text", "").strip()
        question_data = OPT_IN_QUESTIONS[question_index]
        session["answers"][question_data["id"]] = answer
        logger.info(f"User {user_id} answered question {question_index} ('{question_data['id']}'): '{answer}'")

        session["current_question_index"] += 1
        logger.info(f"Incremented question_index for user {user_id} to: {session['current_question_index']}")

        if session["current_question_index"] < len(OPT_IN_QUESTIONS):
            logger.info(f"User {user_id} has more questions. Total questions: {len(OPT_IN_QUESTIONS)}. Asking next.")
            ask_question(user_id, client) # ask_question will now use session["thread_ts"]
        else:
            logger.info(f"User {user_id} has answered all {len(OPT_IN_QUESTIONS)} questions. Proceeding to finalize.")
            try:
                payload_processing = {"channel": user_id, "text": "Great, that's all the questions! Let me process your profile..."}
                if thread_ts_to_use: payload_processing["thread_ts"] = thread_ts_to_use
                client.chat_postMessage(**payload_processing)
                
                profile_data = session["answers"]
                logger.info(f"Collected answers for {user_id}: {profile_data}")

                # Concatenate text for embedding
                text_for_embedding_parts = [
                    profile_data.get("skillset", ""),
                    profile_data.get("background_summary", ""),
                    profile_data.get("current_projects", ""),
                    profile_data.get("topics_of_interest", ""),
                    profile_data.get("networking_goals", ""),
                    profile_data.get("industry", "")
                ]
                text_for_embedding = ". ".join(filter(None, text_for_embedding_parts))
                logger.info(f"Text for embedding for {user_id}: '{text_for_embedding[:200]}...'") # Log snippet
                
                embedding_vector = None
                if text_for_embedding.strip():
                    embedding_vector = get_embedding(text_for_embedding)
                    if embedding_vector:
                        logger.info(f"Successfully generated embedding for {user_id}.")
                    else:
                        logger.error(f"Failed to generate embedding for {user_id} despite having text.")
                else:
                    logger.warning(f"User {user_id} profile has no text content for embedding. Embedding will be null.")

                if embedding_vector is None and text_for_embedding.strip():
                    client.chat_postMessage(channel=user_id, text="Sorry, there was an issue generating insights from your profile. Your text answers have been saved, but you might need to re-opt-in or contact support if search doesn't work as expected.")
                
                data_to_upsert = {
                    "slack_user_id": user_id,
                    **profile_data,
                    "profile_embedding": embedding_vector,
                    "updated_at": "now()"
                }
                logger.info(f"Attempting to upsert data to Supabase for {user_id}.")
                supabase.table("pesto_profiles").upsert(data_to_upsert).execute()
                
                payload_success = {"channel": user_id, "text": "Thanks for opting into Pesto! Your profile has been successfully set up."}
                if thread_ts_to_use: payload_success["thread_ts"] = thread_ts_to_use
                client.chat_postMessage(**payload_success)
                logger.info(f"Successfully saved profile for user {user_id} to Supabase.")

            except Exception as e:
                logger.error(f"Error finalizing profile for user {user_id}: {e}", exc_info=True)
                try:
                    payload_error = {"channel": user_id, "text": "Sorry, there was an error setting up your Pesto profile. Please try again or use /opt-in to restart."}
                    if thread_ts_to_use: payload_error["thread_ts"] = thread_ts_to_use
                    client.chat_postMessage(**payload_error)
                except Exception as slack_e:
                    logger.error(f"Failed to send error DM to user {user_id}: {slack_e}", exc_info=True)
            finally:
                logger.info(f"Cleaning up opt-in session for user {user_id}.")
                if user_id in active_opt_ins: 
                    del active_opt_ins[user_id]
                logger.info(f"Session for {user_id} removed. Current sessions: {list(active_opt_ins.keys())}")
    else:
        logger.warning(f"User {user_id} sent a message, but their question_index ({question_index}) indicates they might have already completed the flow. Ignoring.")

# --- /find Command Handler ---
@app.command("/find")
def handle_find_command(ack, command, client, logger, respond):
    """Handles the /find slash command to search for relevant profiles."""
    ack()
    
    query_text = command.get("text", "").strip()
    querier_user_id = command["user_id"]
    # respond is available for slash commands to send ephemeral or in_channel messages

    if not query_text:
        respond(
            text="Please provide a query to search for. Usage: `/find <your query>`",
            response_type="ephemeral" # Only visible to the user who typed the command
        )
        return

    logger.info(f"User {querier_user_id} initiated /find with query: '{query_text}'")

    # 1. Generate embedding for the query text
    query_embedding = get_embedding(query_text)

    if query_embedding is None:
        logger.error(f"Failed to generate embedding for query: '{query_text}'")
        respond(
            text="Sorry, I couldn't process your query to generate an embedding. Please try again.",
            response_type="ephemeral"
        )
        return
    
    logger.info(f"Successfully generated embedding for query: '{query_text}'")

    # 2. Call Supabase RPC function to find matching profiles
    match_threshold = 0.7  # Adjust as needed (0.0 to 1.0)
    match_count = 3       # Number of top matches to return

    try:
        logger.info(f"Calling RPC match_pesto_profiles with threshold: {match_threshold}, count: {match_count}")
        rpc_response = supabase.rpc(
            "match_pesto_profiles",
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
                "requesting_user_id": querier_user_id
            }
        ).execute()

        if rpc_response.data:
            matched_profiles = rpc_response.data
            logger.info(f"Found {len(matched_profiles)} matching profiles for query '{query_text}'.")
            
            # 3. Format and send the results
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Here are some profiles that might match your query: *\"{query_text}\"*"
                    }
                },
                {"type": "divider"}
            ]

            for profile in matched_profiles:
                profile_info = f"*<@{profile['slack_user_id']}>* ({profile.get('full_name', 'N/A')})\n"
                if profile.get('linkedin_url'):
                    profile_info += f"LinkedIn: <{profile['linkedin_url']}|View Profile>\n"
                if profile.get('skillset'):
                    profile_info += f"Skills: _{profile['skillset']}_\n"
                if profile.get('topics_of_interest'):
                    profile_info += f"Interests: _{profile['topics_of_interest']}_\n"
                # Add more fields if desired, e.g., background_summary, current_projects
                profile_info += f"Similarity: {profile['similarity']:.2f}\n"
                
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": profile_info}
                })
                blocks.append({"type": "divider"})
            
            respond(blocks=blocks, response_type="ephemeral") # Or "in_channel"

        else:
            logger.info(f"No matching profiles found for query: '{query_text}'")
            respond(
                text=f"Sorry, I couldn't find any profiles matching your query: '{query_text}'",
                response_type="ephemeral"
            )

    except Exception as e:
        logger.error(f"Error calling Supabase RPC or processing /find for query '{query_text}': {e}", exc_info=True)
        respond(
            text="Sorry, something went wrong while searching for profiles. Please try again later.",
            response_type="ephemeral"
        )

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start() 