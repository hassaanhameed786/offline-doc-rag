"""Interactive CLI: ask questions about your documents."""
import os

from dotenv import load_dotenv

from rag import Retriever, generate_answer

load_dotenv()


def main() -> None:
    retriever = Retriever()
    use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
    if not use_llm:
        print("(No ANTHROPIC_API_KEY found - showing retrieved chunks only.)")
    print("Ask a question (type 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        hits = retriever.search(question)
        if use_llm:
            print(f"\nAnswer: {generate_answer(question, hits)}\n")
        print("Sources:")
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h['source']} (score {h['score']:.2f}): {h['text'][:100]}...")
        print()


if __name__ == "__main__":
    main()
