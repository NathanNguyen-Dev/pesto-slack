import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedding(text: str, model="text-embedding-ada-002", openai_api_key: str = None) -> list[float]:
    """Generates an embedding for the given text using OpenAI."""
    if not text or not text.strip():
        logger.warning("No text provided for embedding, returning None.")
        return None
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.embeddings.create(
            input=[text.strip()],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding for text '{text[:50]}...': {e}", exc_info=True)
        return None

def batch_update_embeddings():
    """Fetches all profiles from Supabase, re-generates embeddings, and updates them."""
    load_dotenv()

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not all([supabase_url, supabase_key, openai_key]):
        logger.error("Supabase URL/Key or OpenAI API Key is missing from .env file. Exiting.")
        return

    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("Successfully connected to Supabase.")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return

    try:
        response = supabase.table("pesto_profiles").select("slack_user_id, skillset, background_summary, current_projects, topics_of_interest, networking_goals, industry").execute()
        
        if not response.data:
            logger.info("No profiles found in pesto_profiles table.")
            return
        
        profiles = response.data
        logger.info(f"Fetched {len(profiles)} profiles from Supabase.")

    except Exception as e:
        logger.error(f"Error fetching profiles from Supabase: {e}")
        return

    updated_count = 0
    failed_count = 0

    for profile in profiles:
        slack_user_id = profile.get("slack_user_id")
        logger.info(f"Processing profile for user: {slack_user_id}")

        text_parts = [
            profile.get("skillset", ""),
            profile.get("background_summary", ""),
            profile.get("current_projects", ""),
            profile.get("topics_of_interest", ""),
            profile.get("networking_goals", ""),
            profile.get("industry", "")
        ]
        text_for_embedding = ". ".join(filter(None, [part.strip() if part else "" for part in text_parts]))

        if not text_for_embedding.strip():
            logger.warning(f"No content to embed for user {slack_user_id}. Skipping embedding update.")
            continue

        logger.info(f"Generating embedding for user {slack_user_id} with text: '{text_for_embedding[:100]}...'")
        new_embedding = get_embedding(text_for_embedding, openai_api_key=openai_key)

        if new_embedding:
            try:
                supabase.table("pesto_profiles").update({"profile_embedding": new_embedding, "updated_at": "now()"}).eq("slack_user_id", slack_user_id).execute()
                logger.info(f"Successfully updated embedding for user {slack_user_id}.")
                updated_count += 1
            except Exception as e:
                logger.error(f"Error updating embedding in Supabase for user {slack_user_id}: {e}")
                failed_count += 1
        else:
            logger.error(f"Failed to generate new embedding for user {slack_user_id}. Skipping update.")
            failed_count += 1
        
    logger.info(f"--- Batch Embedding Update Complete ---")
    logger.info(f"Successfully updated embeddings for {updated_count} profiles.")
    logger.info(f"Failed to update embeddings for {failed_count} profiles.")

if __name__ == "__main__":
    batch_update_embeddings() 