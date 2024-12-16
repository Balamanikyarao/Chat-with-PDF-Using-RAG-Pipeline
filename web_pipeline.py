from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
import faiss
import openai

# Function to scrape website content
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

# Chunk text into smaller sections
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Create vector embeddings and store in FAISS
def store_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

# Retrieve relevant chunks based on query
def query_website(index, chunks, query, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    _, results = index.search(query_embedding, k=5)
    return [chunks[i] for i in results[0]]

# Generate response using OpenAI GPT
def generate_response(chunks, query):
    context = "\n".join(chunks)
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    return response.choices[0].text.strip()

# Example usage
if __name__ == "__main__":
    # Scrape and process website
    url = "https://www.stanford.edu/"  # Replace with your target website URL
    content = scrape_website(url)
    chunks = chunk_text(content)
    index, chunks = store_embeddings(chunks)

    # Query and generate response
    query = "What programs does Stanford University offer?"  # Replace with your query
    relevant_chunks = query_website(index, chunks, query)
    answer = generate_response(relevant_chunks, query)
    print("Answer:", answer)