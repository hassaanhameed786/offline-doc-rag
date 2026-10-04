"""Interactive CLI: ask questions about your documents."""
import os

from rag import LLM_PROVIDER, OLLAMA_MODEL, Retriever, generate_answer


def llm_available() -> bool:
    if LLM_PROVIDER == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return True  # Ollama needs no API key


def main() -> None:
    retriever = Retriever()
    use_llm = llm_available()
    if use_llm:
        model = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else "claude"
        print(f"Using {LLM_PROVIDER} ({model})")
    else:
        print("(LLM_PROVIDER=anthropic but no ANTHROPIC_API_KEY - showing chunks only.)")
    print("Ask a question (type 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        hits = retriever.search(question)
        if use_llm:
            try:
                print(f"\nAnswer: {generate_answer(question, hits)}\n")
            except Exception as e:
                print(f"\n[LLM error: {e}]")
                print("If using Ollama, make sure it is running: ollama serve\n")
        print("Sources:")
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h['source']} (score {h['score']:.2f}): {h['text'][:300]}...")
        print()


if __name__ == "__main__":
    main()