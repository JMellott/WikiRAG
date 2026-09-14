import os
import json
from dataclasses import dataclass
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass
from google import genai
from google.genai import types

load_dotenv(find_dotenv())

try:
    from config import LLMConfig
    from retriever import RetrievedDocument
except ImportError:
    from backend.config import LLMConfig
    from backend.retriever import RetrievedDocument

import time

@dataclass
class QueryPlan:
    search_terms: list[str]
    mentioned_entities: list[str]

def _interruptible_sleep(seconds: float):
    """Sleeps in 100ms increments so Ctrl+C (KeyboardInterrupt) is processed immediately."""
    end = time.time() + seconds
    while time.time() < end:
        remaining = end - time.time()
        time.sleep(min(0.1, max(0.01, remaining)))

def _execute_with_retry(fn, max_retries: int = 2, initial_delay: float = 2.0):
    """Executes a callable with short backoff on transient 503 / 429 server errors."""
    delay = initial_delay
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            err_msg = str(e)
            is_transient = any(
                code in err_msg
                for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "deadline", "temporarily"]
            )
            if is_transient and attempt < max_retries:
                print(
                    f"[!] Gemini API temporary load spike ({e.__class__.__name__}). "
                    f"Retrying in {delay:.1f}s (Attempt {attempt}/{max_retries}, Press Ctrl+C to abort)...",
                    flush=True,
                )
                try:
                    _interruptible_sleep(delay)
                except KeyboardInterrupt:
                    print("\n[!] Aborted by user (Ctrl+C).", flush=True)
                    raise
                delay *= 2.0
            else:
                raise
    if last_err:
        raise last_err

class RagGenerator:

    def __init__(self, config: LLMConfig):
        self.config = config

        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def analyze_question(self, question: str) -> QueryPlan:
        if not question or not question.strip():
            return QueryPlan(search_terms=[], mentioned_entities=[])

        prompt = (
            "You are an expert game search analyzer for a gaming wiki.\n"
            "Analyze the user's natural language question and output a JSON object with two fields:\n"
            "1. 'mentioned_entities': List of specific bosses, monsters, NPCs, raids, or activities explicitly named in the prompt. Expand common acronyms where appropriate. If no specific entity is named, return an empty list.\n"
            "2. 'search_terms': 2 to 4 concise wiki article titles or search phrases for lexical search. CRITICAL: You MUST include each of the mentioned entities as search terms, along with the core recommendation or activity guide.\n\n"
            "Output valid JSON ONLY in this format:\n"
            '{"mentioned_entities": ["Yama", "Tombs of Amascut"], "search_terms": ["Yama", "Tombs of Amascut", "Money making guide/Combat"]}\n\n'
            f"User Question: {question}\n"
            "JSON:"
        )

        try:
            chat = self.client.chats.create(
                model=self.config.model_name,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=100
                )
            )
            resp = _execute_with_retry(lambda: chat.send_message(prompt))
            raw = (resp.text or "").strip()
            data = json.loads(raw)
            entities = [str(e).strip() for e in data.get("mentioned_entities", []) if str(e).strip()]
            terms = [str(t).strip() for t in data.get("search_terms", []) if str(t).strip()]

            # Ensure all mentioned entities are included in search terms
            for ent in reversed(entities):
                if ent not in terms:
                    terms.insert(0, ent)

            valid_terms = [t for t in terms if len(t) > 1]
            return QueryPlan(
                search_terms=valid_terms if valid_terms else [question.strip()],
                mentioned_entities=entities,
            )
        except Exception as e:
            print(f"[!] Warning: Query analysis error ({e.__class__.__name__}: {str(e)[:80]}). Falling back to direct query.")
            return QueryPlan(search_terms=[], mentioned_entities=[])

    def generate_search_queries(self, question: str) -> list[str]:
        plan = self.analyze_question(question)
        return plan.search_terms

    @staticmethod
    def check_entities_in_context(docs: list[RetrievedDocument], entities: list[str]) -> tuple[list[str], list[str]]:
        if not entities:
            return ([], [])
        if not docs:
            return ([], list(entities))

        present = []
        missing = []

        for entity in entities:
            entity_clean = entity.strip().lower()
            found = False
            for d in docs:
                t_lower = d.title.lower()
                e_lower = d.extract[:3000].lower()
                # Check title match or extract mention
                if entity_clean in t_lower or entity_clean in e_lower:
                    found = True
                    break
            if found:
                present.append(entity)
            else:
                missing.append(entity)

        return (present, missing)

    def format_context(
            self, 
            docs: list[RetrievedDocument], 
            max_docs: int = 8, 
            max_total_chars: int = 60000, 
            priority_entities: list[str] | None = None
        ) -> str:
        if not docs:
            return "No matching wiki articles were retrieved."

        ordered_docs = list(docs)
        if priority_entities:
            pe_lower = [e.strip().lower() for e in priority_entities if e.strip()]
            priority_docs = []
            other_docs = []
            for d in ordered_docs:
                t_lower = d.title.lower()
                if any(e in t_lower for e in pe_lower):
                    priority_docs.append(d)
                else:
                    other_docs.append(d)
            ordered_docs = priority_docs + other_docs

        blocks = []
        total_chars = 0
        for i, doc in enumerate(ordered_docs[:max_docs], start = 1):
            content = doc.extract
            budget = max_total_chars - total_chars
            if budget <= 1000:
                break
            if len(content) > budget:
                content = content[:budget] + "\n...[Remaining section truncated]"

            block = (
                f"[Document {i}]\n"
                f"Title: {doc.title}\n"
                f"Source URL: {doc.url}\n"
                f"Content:\n{content}"
            )
            blocks.append(block)
            total_chars += len(content)

        return "\n\n".join(blocks)

    def generate_answer(
            self, 
            question: str, 
            docs: list[RetrievedDocument],
            mentioned_entities: list[str] | None = None
        ) -> str:
        context_block = self.format_context(docs)

        entity_instruction = ""
        if mentioned_entities:
            entity_instruction = (
                f"The user specifically mentioned the following benchmark entities/activities: {mentioned_entities}.\n"
                "1. First, examine the retrieved wiki documents for each of those entities to understand their combat mechanics, difficulty level, and team vs solo scaling.\n"
                "2. Use that information to accurately gauge the user's current PvM capability, gear progression, and mechanical skill ceiling.\n"
                "3. Recommend money-making bosses or activities from the context that match the user's assessed skill bracket, explicitly comparing their difficulty to the entities the user mentioned.\n"
            )

        system_instruction = (
            "You are a helpful, expert video game assistant. "
            "Answer the user's question using the provided wiki context documents below. "
            "If the question cannot be answered at all with the context, explicitly start your response with '[NOT_FOUND]'. "
            f"{entity_instruction}"
            "For recommendation or advice questions (e.g. which boss to kill for money, what training method to use), "
            "recommend appropriate options found in the context that match the user's stated level or difficulty, "
            "and explain why they fit. "
            "Every factual claim and recommendation must cite its source using markdown links in the format [Title](URL) "
            "using the exact URLs provided."
        )

        user_content = (
            f"=== RETRIEVED WIKI CONTEXT ===\n"
            f"{context_block}\n"
            f"==============================\n\n"
            f"User Question: {question}"
        )

        chat = self.client.chats.create(
            model = self.config.model_name,
            config = types.GenerateContentConfig(
                system_instruction = system_instruction,
                temperature = self.config.temperature,
                max_output_tokens = self.config.max_output_tokens,
            )
        )

        response = _execute_with_retry(lambda: chat.send_message(user_content))

        return response.text or "No response generated."

    @staticmethod
    def unable_to_answer(answer: str) -> bool:
        if not answer or not answer.strip():
            return True

        answer = answer.lower().strip()

        has_citations = "http://" in answer or "https://" in answer

        if answer.startswith("[not_found]"):
            if has_citations and len(answer) > 250:
                return False
            return True

        if has_citations and len(answer) > 150:
            return False

        if len(answer) < 250:
            indicators = [
                "[not_found]",
                "could not find",
                "couldn't find",
                "cannot find",
                "can't find",
                "not found in the provided",
                "not contained in the provided",
                "not mentioned in the provided",
                "not provided in the",
                "does not contain information",
                "no information on",
                "no information about",
            ]

            return any(indicator in answer for indicator in indicators)

        return False

    @staticmethod
    def clean_answer(answer : str) -> str:
        return answer.replace("[NOT_FOUND]", "").strip()