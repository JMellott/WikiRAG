from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

@dataclass
class WikiConfig:
    api_url: str = "https://oldschool.runescape.wiki/api.php"
    search_limit: int = 20
    max_search_limit: int = 100
    limit_step: int = 10
    timeout_seconds: int = 10
    user_agent: str = "WikiRAG/1.0"
    explaintext: bool = True
    exintro: bool = False
    max_chars_per_doc: int = 15000

@dataclass
class LLMConfig:
    model_name: str = "gemini-3.1-flash-lite"
    temperature: float = 0.2
    max_output_tokens: int = 1024

@dataclass
class AppConfig:
    wiki: WikiConfig = field(default_factory=WikiConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def load(cls, config_path: str = "../config.yaml") -> "AppConfig":
        path = Path(config_path)

        data = {}
        if path.exists():
            if path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                except ImportError:
                    pass

        wiki_data = data.get("wiki", {})
        llm_data = data.get("llm", {})

        api_url = os.getenv("WIKI_API_URL", wiki_data.get("api_url", WikiConfig.api_url))
        search_limit = int(os.getenv("WIKI_SEARCH_LIMIT", wiki_data.get("search_limit", WikiConfig.search_limit)))
        max_search_limit = int(os.getenv("WIKI_MAX_SEARCH_LIMIT", wiki_data.get("max_search_limit", WikiConfig.max_search_limit)))
        limit_step = int(os.getenv("WIKI_LIMIT_STEP", wiki_data.get("limit_step", WikiConfig.limit_step)))
        timeout_seconds = int(os.getenv("WIKI_TIMEOUT", wiki_data.get("timeout_seconds", WikiConfig.timeout_seconds)))
        user_agent = os.getenv("WIKI_USER_AGENT", wiki_data.get("user_agent", WikiConfig.user_agent))
        explaintext = wiki_data.get("explaintext", WikiConfig.explaintext)
        exintro = wiki_data.get("exintro", WikiConfig.exintro)
        max_chars_per_doc = int(os.getenv("WIKI_MAX_CHARS_PER_DOC", wiki_data.get("max_chars_per_doc", WikiConfig.max_chars_per_doc)))

        model_name = os.getenv("LLM_MODEL_NAME", llm_data.get("model_name", LLMConfig.model_name))
        temperature = float(os.getenv("LLM_TEMPERATURE", llm_data.get("temperature", LLMConfig.temperature)))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", llm_data.get("max_output_tokens", LLMConfig.max_output_tokens)))

        return cls(
            wiki=WikiConfig(
                api_url=api_url,
                search_limit=search_limit,
                max_search_limit=max_search_limit,
                limit_step = limit_step,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
                max_chars_per_doc=max_chars_per_doc
            ),
            llm=LLMConfig(
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )