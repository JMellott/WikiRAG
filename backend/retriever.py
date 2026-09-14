from dataclasses import dataclass
import re
import requests

try:
    from config import WikiConfig
except ImportError:
    from backend.config import WikiConfig

@dataclass
class RetrievedDocument:
    page_id: int
    title: str
    url: str
    extract: str

_nlp = None

def get_spacy_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

EXTRA_SEARCH_STOPWORDS = {
    # Actions / verbs
    "find", "obtain", "kill", "beat", "defeat", "tell", "need", "want", "learn", "acquire", "use", "using", "give",
    # Evaluation, formatting, comparison, and meta-intent terms
    "pro", "pros", "con", "cons", "advantage", "advantages", "disadvantage", "disadvantages",
    "benefit", "benefits", "drawback", "drawbacks", "difference", "differences",
    "compare", "comparison", "versus", "vs", "different", "various", "multiple",
    # Qualifiers & superlatives
    "best", "top", "worst", "fast", "fastest", "slow", "slowest", "cheap", "cheapest", "expensive",
    "good", "better", "great", "optimal", "favorite", "popular", "recommended",
    # Method / guide descriptors
    "method", "methods", "way", "ways", "option", "options", "strategy", "strategies",
    "list", "guide", "overview", "summary", "explanation", "explain", "detail", "details",
    # Progression / level constraints
    "level", "levels", "tier", "tiers", "stat", "stats",
    "after", "before", "around", "above", "below", "between"
}

def clean_query(query: str) -> str:
    cleaned = query.strip()
    if not cleaned:
        return ""

    try:
        nlp = get_spacy_nlp()
        doc = nlp(cleaned)
        
        keywords = [
            token.text for token in doc
            if not token.is_stop
            and token.lemma_.lower() not in EXTRA_SEARCH_STOPWORDS
            and token.text.lower() not in EXTRA_SEARCH_STOPWORDS
            and not token.is_punct
            and not token.is_space
            and not token.is_digit
        ]
        
        extracted = " ".join(keywords).strip()
        return extracted if extracted else cleaned
    except Exception:
        pass

    words = re.sub(r'[\?\.!,;]+', ' ', cleaned).split()
    basic_stop = {
        "how", "do", "i", "can", "to", "get", "a", "an", "the", "where",
        "is", "are", "what", "for", "in", "of", "on", "with"
    }
    filtered = [
        w for w in words 
        if w.lower() not in basic_stop
        and w.lower() not in EXTRA_SEARCH_STOPWORDS
        and not w.isdigit()
    ]
    return " ".join(filtered) if filtered else cleaned

class WikiRetriever:

    def __init__(self, config: WikiConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config.user_agent,
            "Accept": "application/json",
        })

    def search(self, query: str, limit: int | None = None, search_terms: list[str] | None = None) -> list[RetrievedDocument]:
        if not query or not query.strip():
            return []

        search_limit = limit if limit is not None else self.config.search_limit
        if search_terms:
            return self.search_multi(search_terms, search_limit)
        
        search_query = clean_query(query)

        docs = self._query_mediawiki(search_query, search_limit)
        
        # Fallback to raw query if keyword extraction returned nothing
        if not docs and search_query.lower() != query.strip().lower():
            docs = self._query_mediawiki(query.strip(), search_limit)

        return docs

    def search_multi(self, queries: list[str], limit: int | None = None):
        search_limit = limit if limit is not None else self.config.search_limit
        seen_page_ids: set[int] = set()
        aggregated: list[RetrievedDocument] = []

        for query in queries:
            if not query or not query.strip():
                continue

            cleaned = clean_query(query)
            term_to_search = cleaned if cleaned else query.strip()
            found = self._query_mediawiki(term_to_search, search_limit)

            if not found and term_to_search != query.strip():
                found = self._query_mediawiki(query.strip(), search_limit)

            for doc in found:
                if doc.page_id not in seen_page_ids:
                    seen_page_ids.add(doc.page_id)
                    aggregated.append(doc)

            if len(aggregated) >= search_limit:
                break

        return aggregated[:search_limit]

    def _fetch_page_extract(self, title: str) -> str:
        if not title:
            return ""
        try:
            params = {
                "action": "query",
                "prop": "extracts",
                "titles": title,
                "explaintext": "1" if self.config.explaintext else "0",
                "format": "json",
                "formatversion": "2",
            }
            res = self.session.get(
                self.config.api_url,
                params=params,
                timeout=self.config.timeout_seconds,
            )
            if res.ok:
                pages = res.json().get("query", {}).get("pages", [])
                if pages:
                    extract = pages[0].get("extract", "").strip()
                    max_chars = getattr(self.config, "max_chars_per_doc", 15000)
                    if max_chars and len(extract) > max_chars:
                        return extract[:max_chars] + "\n...[Content truncated for length]"
                    return extract
        except Exception:
            pass
        return ""

    def _query_mediawiki(self, search_term: str, search_limit: int) -> list[RetrievedDocument]:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": search_term,
            "gsrlimit": str(search_limit),
            "prop": "extracts|info" if self.config.exintro else "info",
            "inprop": "url",
            "explaintext": "1" if self.config.explaintext else "0",
            "format": "json",
            "formatversion": "2",
        }
        if self.config.exintro:
            params["exintro"] = "1"

        data = {}
        for attempt in range(3):
            response = self.session.get(
                        self.config.api_url,
                        params=params,
                        timeout=self.config.timeout_seconds,
            )
            if response.status_code == 429:
                wait_sec = float(response.headers.get("Retry-After", 2.0 * (attempt + 1)))
                print(f"[!] MediaWiki API rate limit (429). Backing off for {wait_sec:.1f}s (Attempt {attempt+1}/3)...", flush=True)
                time.sleep(wait_sec)
                continue
            response.raise_for_status()
            data = response.json()
            break

        pages = data.get("query", {}).get("pages", [])
        
        sorted_pages = sorted(pages, key=lambda p: p.get("index", 999))

        if not self.config.exintro:
            from concurrent.futures import ThreadPoolExecutor
            pages_to_fetch = [p for p in sorted_pages if not p.get("extract") and not p.get("missing")]
            if pages_to_fetch:
                with ThreadPoolExecutor(max_workers=min(5, len(pages_to_fetch) or 1)) as executor:
                    fetched = list(executor.map(lambda p: self._fetch_page_extract(p.get("title", "")), pages_to_fetch))
                for p, ext in zip(pages_to_fetch, fetched):
                    p["extract"] = ext

        return [
            RetrievedDocument(
                page_id=p.get("pageid", 0),
                title=p.get("title", "Untitled"),
                url=p.get("fullurl") or p.get("canonicalurl", ""),
                extract=p.get("extract", "").strip(),
            )
            for p in sorted_pages
            if not p.get("missing") and p.get("extract", "").strip()
        ]