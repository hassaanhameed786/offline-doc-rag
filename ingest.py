"""Build the vector index from files in ./data"""
import argparse

from rag import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents for RAG")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    n = build_index(args.data_dir)
    print(f"Indexed {n} chunks into ./index")
