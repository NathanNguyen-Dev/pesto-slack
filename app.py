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

    try:
        # Check if user already exists in Supabase
        response = supabase.table("pesto_profiles").select("slack_user_id").eq("slack_user_id", user_id).execute()
        if response.data:
            logger.info(f"User {user_id} attempted /opt-in but already exists in Supabase.")
            client.chat_postMessage(
                channel=user_id,
                text="You've already opted into Pesto! If you'd like to update your profile, please use the `/re-opt-in` command."
            )
            # Also send a confirmation to the channel where the command was invoked, if it's a public channel
            # This is good practice so the user knows the bot responded, even if it's just to say "check DMs" or "already in"
            if body.get("channel_id") != user_id: # i.e. not a DM with the bot already
                 client.chat_postMessage(
                    channel=body["channel_id"],
                    text=f"<@{user_id}>, you're already opted in! I've sent you a DM with more details."
                )
            return
    except Exception as e:
        logger.error(f"Error checking Supabase for existing user {user_id} during /opt-in: {e}", exc_info=True)
        try:
            client.chat_postMessage(
                channel=user_id,
                text="Sorry, I couldn't check your current status. Please try again in a moment."
            )
        except Exception as slack_e:
            logger.error(f"Failed to send error DM to user {user_id}: {slack_e}")
        return

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

# --- /re-opt-in Command Handler ---
@app.command("/re-opt-in")
def handle_re_opt_in_command(ack, body, client, logger):
    """Handles the /re-opt-in slash command to allow users to update their profile."""
    ack()
    user_id = body["user_id"]
    logger.info(f"User {user_id} initiated /re-opt-in.")

    # Clear any existing session for this user, if one somehow exists, to ensure a fresh start
    if user_id in active_opt_ins:
        logger.info(f"Clearing pre-existing active_opt_ins session for user {user_id} before /re-opt-in.")
        del active_opt_ins[user_id]

    # Start a new opt-in session
    active_opt_ins[user_id] = {
        "current_question_index": 0,
        "answers": {},
        "thread_ts": None # Initialize thread_ts for the new session
    }
    logger.info(f"Initialized new opt-in session for user {user_id} via /re-opt-in.")

    try:
        # Notify in the channel where command was invoked (if not a DM)
        if body.get("channel_id") != user_id: # i.e. not a DM with the bot already
            client.chat_postMessage(
                channel=body["channel_id"],
                text=f"<@{user_id}>, I\'ll send you a DM to update your Pesto profile! Please check your DMs."
            )
        
        # Send initial DM
        client.chat_postMessage(
            channel=user_id,
            text="Hi there! You've requested to update your Pesto profile. Let\'s go through the questions again."
            # First message in DM is not threaded
        )
        ask_question(user_id, client) # Ask the first question
        logger.info(f"Started re-opt-in question flow for user {user_id}.")
    except Exception as e:
        logger.error(f"Error starting re-opt-in conversation with user {user_id}: {e}", exc_info=True)
        try:
            client.chat_postMessage(
                channel=user_id,
                text="Sorry, I couldn\'t start the profile update process. Please try again in a moment."
            )
        except Exception as slack_e:
            logger.error(f"Failed to send error DM to user {user_id} during re-opt-in setup: {slack_e}")

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
    
    original_query_text = command.get("text", "").strip()
    querier_user_id = command["user_id"]

    if not original_query_text:
        respond(
            text="Please provide a query to search for. Usage: `/find <your query>`",
            response_type="ephemeral"
        )
        return

    logger.info(f"User {querier_user_id} initiated /find with original query: '{original_query_text}'")

    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # 1. Extract keywords using an LLM
    extracted_keywords_or_phrase = "" 
    try:
        # Option 2.1: More explicit instructions and few-shot examples
        keyword_extraction_system_prompt = """You are an expert at extracting key entities and concepts from a user's query for finding matching professional profiles. Your goal is to distill the query into its most essential components: core skills, technologies, job functions, industries, or primary professional interests.
Instructions:
1. Identify the core subjects or entities the user is looking for.
2. Remove stop words, articles, and conversational filler (e.g., 'find me a', 'looking for', 'who is', 'working on').
3. Do NOT include generic nouns like 'person', 'individual', 'someone' if more specific professional descriptors or concepts are present or implied. For example, 'business person' should be refined to 'business' or 'business professional'.
4. Return a concise, comma-separated list of these core keywords or a very short descriptive phrase.
Example Query: 'find me a business person working on Robotic'
Example Output: 'business, Robotic'
Example Query: 'someone skilled in python and data analysis'
Example Output: 'Python, data analysis'"""
        
        prompt_messages_keyword_extraction = [
            {"role": "system", "content": keyword_extraction_system_prompt},
            {"role": "user", "content": f"User query: {original_query_text}"}
        ]
        
        logger.info(f"Sending query to LLM for keyword/search phrase extraction (Prompt v2.1): {original_query_text}")
        chat_completion_response = openai_client.chat.completions.create(
            model="gpt-4o-mini", # Updated model
            messages=prompt_messages_keyword_extraction,
            temperature=0.1 # Very low temperature for more precise adherence to examples
        )
        text_to_embed = chat_completion_response.choices[0].message.content.strip()
        logger.info(f"LLM extracted keywords/search phrase (v2.1): '{text_to_embed}' from original query: '{original_query_text}'")

        if not text_to_embed:
            logger.warning(f"LLM (v2.1) did not return keywords/search phrase for query: '{original_query_text}'. Using original query for embedding.")
            text_to_embed = original_query_text

    except Exception as e:
        logger.error(f"Error calling LLM (v2.1) for keyword/search phrase extraction for query '{original_query_text}': {e}", exc_info=True)
        respond(text="Sorry, I had trouble refining your query. Please try again.", response_type="ephemeral")
        return

    # 2. Generate embedding
    query_embedding = get_embedding(text_to_embed)
    if query_embedding is None:
        logger.error(f"Failed to generate embedding for text: '{text_to_embed}'")
        respond(text="Sorry, I couldn't process your query for embedding. Please try again.", response_type="ephemeral")
        return
    logger.info(f"Successfully generated embedding for text: '{text_to_embed}'")

    # 3. Call Supabase RPC
    match_threshold = 0.7  
    match_count = 3 
    try:
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
            logger.info(f"Found {len(matched_profiles)} matching profiles for refined query text '{text_to_embed}' (original: '{original_query_text}').")
            
            # 4. Generate explanations for all matches in a single LLM call
            all_explanations = ["Could not generate explanation for this match." for _ in matched_profiles]
            if matched_profiles: # Only call LLM if there are profiles to explain
                profile_summaries_for_llm = []
                for i, profile in enumerate(matched_profiles):
                    parts = [
                        f"Profile {i+1}:",
                        f"Name: {profile.get('full_name', 'N/A')}",
                        f"Skills: {profile.get('skillset', 'N/A')}",
                        f"Background: {profile.get('background_summary', 'N/A')}",
                        f"Interests: {profile.get('topics_of_interest', 'N/A')}",
                        f"Industry: {profile.get('industry', 'N/A')}"
                    ]
                    profile_summaries_for_llm.append(" ".join(parts))
                
                combined_profile_text = "\n\n".join(profile_summaries_for_llm)

                explanation_prompt_messages = [
                    {"role": "system", "content": f"You are an expert at explaining why professional profiles are good matches for a user's query. For each profile provided below, write a concise 1-2 sentence explanation. Start each explanation with 'Profile X is a good match because...' or similar, ensuring each explanation is clearly for the corresponding profile number. Output each explanation on a new line, numbered. Example: 1. Explanation for profile 1. 2. Explanation for profile 2."},
                    {"role": "user", "content": f"User query: \"{original_query_text}\".\n\nMatched profiles:\n{combined_profile_text}\n\nProvide a numbered list of explanations, one for each profile, explaining why it matches the user query."}
                ]
                try:
                    logger.info(f"Generating explanations for {len(matched_profiles)} profiles based on query '{original_query_text}'")
                    explanation_response = openai_client.chat.completions.create(
                        model="gpt-4o-mini", # Updated model
                        messages=explanation_prompt_messages,
                        temperature=0.3
                    )
                    raw_explanations_text = explanation_response.choices[0].message.content.strip()
                    logger.info(f"LLM raw explanations: '{raw_explanations_text}'")
                    
                    # Attempt to parse numbered list of explanations
                    parsed_explanations = []
                    for line in raw_explanations_text.split('\n'):
                        line = line.strip()
                        if line and (line[0].isdigit() and (line[1] == '.' or line[1:3] == '. ')):
                            parsed_explanations.append(line.split('.', 1)[1].strip() if '.' in line else line.split(' ',1)[1].strip() ) 
                    
                    if len(parsed_explanations) == len(matched_profiles):
                        all_explanations = parsed_explanations
                    else:
                        logger.warning(f"Could not parse explanations correctly. Expected {len(matched_profiles)}, got {len(parsed_explanations)}. Using default.")
                        # Fallback: we could try to assign first N explanations if count is off, or just use default.

                except Exception as e:
                    logger.error(f"Error generating LLM explanations for profiles: {e}", exc_info=True)
            
            # 5. Format and send the results
            response_blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Based on your query *\"{original_query_text}\"* (refined to: *\"{text_to_embed}\"*), here are some profiles that might match:"
                    }
                },
                {"type": "divider"}
            ]

            for i, profile in enumerate(matched_profiles):
                explanation = all_explanations[i]
                profile_info_blocks = [
                    f"*<@{profile['slack_user_id']}>* ({profile.get('full_name', 'N/A')})",
                    f"*Reason for match:* {explanation}",
                    f"Similarity Score: {profile['similarity']:.2f}"
                ]
                if profile.get('linkedin_url'): profile_info_blocks.append(f"LinkedIn: <{profile['linkedin_url']}|View Profile>")
                if profile.get('skillset'): profile_info_blocks.append(f"Skills: _{profile['skillset']}_")
                if profile.get('topics_of_interest'): profile_info_blocks.append(f"Interests: _{profile['topics_of_interest']}_")
                
                response_blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "\n".join(profile_info_blocks)}
                })
                response_blocks.append({"type": "divider"})
            
            respond(blocks=response_blocks, response_type="ephemeral")

        else:
            logger.info(f"No matching profiles found for refined query text: '{text_to_embed}' (original: '{original_query_text}').")
            respond(text=f"Sorry, I couldn't find any profiles matching your query: '{original_query_text}' (refined to: '{text_to_embed}').", response_type="ephemeral")

    except Exception as e:
        logger.error(f"Error calling Supabase RPC or processing /find for refined query '{text_to_embed}': {e}", exc_info=True)
        respond(text="Sorry, something went wrong while searching for profiles. Please try again later.", response_type="ephemeral")

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start() 