# About Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a retrieval system with a large language model.
Instead of relying only on what the model memorized during training, RAG first searches a
knowledge base for relevant passages and then gives those passages to the model as context.

## Why use RAG?

RAG reduces hallucinations because answers are grounded in real documents. It also lets you
use private or up-to-date data without retraining or fine-tuning a model. Sources can be cited,
which makes answers easier to verify.

## Typical pipeline

1. Load documents (PDFs, markdown, web pages).
2. Split them into overlapping chunks so each piece fits the embedding model.
3. Convert each chunk into a vector using an embedding model.
4. At question time, embed the query and find the most similar chunks using cosine similarity.
5. Send the top chunks plus the question to an LLM and ask it to answer using only that context.

## Common improvements

Hybrid search (keyword + vector), re-ranking retrieved chunks, metadata filtering, and
evaluation with question/answer test sets all improve quality in production systems.
