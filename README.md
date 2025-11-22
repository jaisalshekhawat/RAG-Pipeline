**📌 Overview**

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline using:

LangChain for text chunking

SentenceTransformers for embeddings

FAISS for vector search

Hugging Face Transformers (TinyLlama) for answer generation

**The pipeline processes your documents end-to-end:**

Load → Clean → Split → Embed → Store → Retrieve → Generate Answer


Everything is done using pure Python scripts with minimal dependencies and clear structure.

**📂 Project Structure:**
RAG-pipeline/
│
├── main.py                # Runs the entire RAG workflow end-to-end
│
├── prepare_data.py        # Loads and cleans raw documents
├── load_data.py           # Reads .txt files from the data folder
├── clean_data.py          # Removes extra whitespace + non-ASCII chars
│
├── split_text.py          # Splits documents into text chunks
│
├── create_embeddings.py   # Generates embeddings using MiniLM
├── store_faiss.py         # Creates FAISS index + saves metadata
│
├── retrieve_faiss.py      # Loads index + retrieves similar chunks
├── generate_answer.py     # Uses TinyLlama to generate final answers
│
└── data/                  # Place your .txt files here

**🚀 How the Pipeline Works**
1️⃣ Load & Clean Documents

prepare_docs()

Loads .txt files

Cleans text using regex

Returns cleaned documents

2️⃣ Split Documents into Chunks

split_docs()

Uses RecursiveCharacterTextSplitter

Splits text into ~500-character chunks with 100 overlap

Produces LangChain Document objects

3️⃣ Generate Embeddings

get_embeddings()

Loads all-MiniLM-L6-v2

Encodes each chunk into a vector

Returns a NumPy embedding matrix

4️⃣ Store Embeddings in FAISS

build_faiss_index()

Creates a IndexFlatL2 FAISS index

Adds embeddings

Saves the index to disk

Metadata saved via pickle (save_metadata())

5️⃣ Retrieve Similar Chunks

retrieve_similar_chunks()

Re-embeds the query

Searches FAISS

Returns the top-k text chunks

6️⃣ Generate Final Answer

generate_answer()

Loads TinyLlama model

Builds a prompt with:

Context:
<retrieved chunks>

Question:
<your question>


Generates the final response

▶️ Run the Pipeline

Make sure your .txt files are inside the data/ directory.

**Run:**

python main.py


The pipeline will:

Load and clean documents

Chunk them

Build embeddings

Create a FAISS index

Retrieve relevant chunks

Generate an LLM answer

You will see output like:

Total chunks created: 57
Embeddings shape: (57, 384)
Stored embeddings and metadata successfully.
Final Answer:
<model output>

📄 Example Query

Inside main.py, the pipeline runs:

query = "Does unsupervised ML cover regression tasks?"
generate_answer(query)


You can change this to any question.

**🛠️ Dependencies**

You will need:
  faiss-cpu
  numpy
  langchain
  sentence-transformers
  transformers
  torch


💡 Notes

The number of chunks is determined by chunk_size & chunk_overlap, not manually set.

FAISS index and metadata must match the order of embeddings.

TinyLlama loads every time generate_answer runs (intended in your code).

Chunking uses sentence/paragraph boundaries (RecursiveCharacter strategy).

