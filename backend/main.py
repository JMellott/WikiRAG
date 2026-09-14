import argparse
import sys
from config import AppConfig
from retriever import WikiRetriever
from generator import RagGenerator
import time

def run_pipeline(question: str, config_path: str = "../config.yaml"):
    config = AppConfig.load(config_path)
    print(f"[*] Target Wiki: {config.wiki.api_url}")
    print(f"[*] Initial Search Limit: {config.wiki.search_limit} documents (Max: {config.wiki.max_search_limit})")
    print(f"[*] Question: {question}\n")

    retriever = WikiRetriever(config.wiki)
    generator = RagGenerator(config.llm)

    current_limit = config.wiki.search_limit
    max_limit = config.wiki.max_search_limit
    step = config.wiki.limit_step

    query_plan = generator.analyze_question(question)
    optimized_terms = query_plan.search_terms
    mentioned_entities = query_plan.mentioned_entities

    if mentioned_entities:
        print(f"[*] Specifically Mentioned Entities / Benchmarks: {mentioned_entities}")
    if optimized_terms:
        print(f"[*] Optimized Wiki Search Terms: {optimized_terms}")
    else:
        from retriever import clean_search_query
        term = clean_search_query(question)
        print(f"[*] Search Terms (NLP Cleaned): '{term}'")

    attempt = 1
    docs = []
    answer = ""

    while True:
        print(f"[*] [Attempt {attempt}] Querying MediaWiki API (limit={current_limit})...")
        docs = retriever.search(question, limit=current_limit, search_terms=optimized_terms)
        print(f"[+] Retrieved {len(docs)} document(s):")
        for d in docs:
            print(f"    - {d.title}: {d.url}")

        if not docs:
            print("[!] No matching documents found on the wiki.")
            answer = "I could not find any relevant wiki articles to answer your question."
            break

        if mentioned_entities:
            present, missing = generator.check_entities_in_context(docs, mentioned_entities)
            if missing:
                print(f"\n[*] Waiting to generate answer: Entity context missing for {missing}. Performing targeted retrieval...")
                existing_ids = {d.page_id for d in docs}
                added_entity_docs = []
                for ent in missing:
                    print(f"    Fetching wiki page for '{ent}' to gauge player ability/difficulty...")
                    ent_docs = retriever.search(ent, limit=2)
                    for ed in ent_docs:
                        if ed.page_id not in existing_ids:
                            existing_ids.add(ed.page_id)
                            added_entity_docs.append(ed)
                            print(f"    [+] Added '{ed.title}' ({ed.url}) to context.")

                # Prepend added entity documents so they are prioritized in LLM context
                docs = added_entity_docs + docs
                present, still_missing = generator.check_entities_in_context(docs, mentioned_entities)
                if present:
                    print(f"[+] Verified entity context in documents: {present}")
                if still_missing:
                    print(f"[!] Note: No wiki articles available for: {still_missing}")

        print(f"\n[*] Generating answer with Gemini using {len(docs)} context document(s)...")
        raw_answer = generator.generate_answer(question, docs)

        if generator.unable_to_answer(raw_answer):
            if current_limit < max_limit and len(docs) >= current_limit:
                next_limit = min(current_limit + step, max_limit)
                print(f"\n[!] Answer was not found in top {len(docs)} document(s).")
                print(f"    Increasing search limit from {current_limit} to {next_limit} and re-querying wiki...\n")
                current_limit = next_limit
                attempt += 1
                time.sleep(1.0)
                continue
            else:
                if len(docs) < current_limit:
                    print(f"[*] Exhausted all available wiki pages matching the query ({len(docs)} total).")
                else:
                    print(f"[*] Reached maximum search limit ({max_limit}).")

        answer = generator.clean_answer(raw_answer)
        break

    print("\n" + "=" * 60)
    print("ASSISTANT RESPONSE:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "MediaWiki Wiki RAG Pipeline")
    parser.add_argument("question", nargs = "?", help = "Question to ask")
    parser.add_argument("--config", default = "../config.yaml", help = "Path to config.yaml")
    args = parser.parse_args()

    run_pipeline(args.question, args.config)