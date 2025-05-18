# Pesto Slack Bot - Development Progress

## Overall Goal
Create an MVP Pesto Slack bot where users can `/opt-in` with their details through a conversational flow and others can `/find` relevant profiles using embedding-based similarity.

## Current Status: Phase 1 (Opt-in Flow) - Largely Complete

We have successfully implemented the conversational opt-in flow. The bot now interacts with users in Direct Messages to collect profile information, and it handles replies in threads correctly.

### Key Features Implemented:

1.  **Project Setup:**
    *   Python project initialized with a virtual environment.
    *   Dependencies (`slack_bolt`, `supabase`, `openai`, `python-dotenv`) managed via `requirements.txt`.
    *   `.env` file for environment variable management.
    *   `supabase_client.py` for Supabase interaction.
    *   Basic `app.py` structure using `slack_bolt` with Socket Mode.

2.  **Supabase Configuration:**
    *   `pesto_profiles` table schema defined for 10 user questions, plus `slack_user_id`, `profile_embedding`, and timestamps.
    *   `pgvector` extension is assumed to be enabled in Supabase.

3.  **Slack App Configuration (Assumed Complete by User):**
    *   Slack app created.
    *   Socket Mode enabled.
    *   Necessary OAuth Scopes added (e.g., `commands`, `chat:write`, `im:history`, `message.im` event subscription).
    *   `/opt-in` slash command registered.

4.  **Conversational `/opt-in` Flow (`app.py`):**
    *   **Initiation:** User types `/opt-in` in any channel.
        *   Bot sends an ephemeral message to the channel (or a public one if ephemeral fails) directing the user to DMs.
        *   Bot starts a DM conversation with the user.
    *   **Question Asking:** Bot asks 10 predefined questions one by one.
    *   **State Management:** An in-memory dictionary (`active_opt_ins`) tracks user progress and collected answers, including `thread_ts` for contextual replies.
    *   **Threaded Replies:** The bot now correctly replies in threads if the user answers in a thread. If the user answers in the main DM, the bot replies in the main DM.
    *   **Data Collection:** All 10 answers are collected.
    *   **Embedding Generation:**
        *   A helper function `get_embedding()` uses the OpenAI API (`text-embedding-ada-002`) to generate embeddings from a concatenation of selected answers (`skillset`, `background_summary`, `current_projects`, `topics_of_interest`, `networking_goals`, `industry`).
    *   **Supabase Storage:**
        *   All 10 answers and the generated `profile_embedding` are upserted into the `pesto_profiles` table in Supabase.
    *   **User Confirmation:** Bot sends a confirmation message upon successful opt-in or an error message if issues occur.
    *   **Logging:** Detailed logging has been added to help with debugging.

### Files Created/Modified:

*   `Plan.md`: Updated to reflect the conversational opt-in flow and 10-question structure.
*   `requirements.txt`: Contains project dependencies.
*   `supabase_client.py`: Initializes the Supabase client.
*   `app.py`: Contains the core bot logic for the `/opt-in` conversational flow, including state management, question asking, OpenAI embedding, Supabase interaction, and threaded reply handling.
*   `.env` (User managed): For storing API keys and secrets.

## Next Steps (Phase 2 from Plan.md):

*   Implement the `/find` command handler.
*   Create the Supabase RPC function (`match_pesto_profiles`) for vector similarity search.
*   Implement logic to call the RPC function from Python.
*   Format and send recommendations back to the user in Slack.
*   Thorough testing of the `/find` functionality.

---
*This document reflects the progress as of the last interaction.* 