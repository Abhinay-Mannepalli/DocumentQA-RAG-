# DocumentQA-RAG-
## Tech stack
- Vector Database: Pinecone
- Embedding model: 'text-embedding-ada-002'
- LLM: gpt-3.5-turbo
- Language Used- Python

## Architechture

<img width="1280" alt="Screenshot 2024-09-12 at 12 59 37 PM" src="https://github.com/user-attachments/assets/bdf8692f-624a-42dc-a2df-856af5d5cb9a">

## Implementation
- Install all the requirements in requirement.txt.
  ```sh
  pip install -r requirements.txt
  
  ```
- Create a OPENAI API User secret key in platform.openai.com
- Create a Pinecone index in pinecone.io
    - Cloud provider: AWS
- Create a '.env' file and include the following keys
    - OPENAI_KEY
    - PINECONE_KEY
    - PINECONE_INDEX
- Run all the cells in vectorDB.ipynb.
    - Reads the data from documents using PYPDF.
    - splits the data into chunks(Customize your chunk size).
    - Generates the embeddings using embedding model

