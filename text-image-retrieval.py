# https://www.interviewquery.com/questions/text-to-image-retrieval?playlist=ai-engineering-50

# You work as a machine learning engineer at Amazon, focusing on product 
# discovery. Your team is tasked with building a system that enables customers 
# to enter a text description, such as “red hiking backpack with water bottle 
# holder”, and retrieve the most relevant product images from Amazon’s vast catalog.
# How would you design this system end-to-end?

# store products with image embeddings -> pgvector + postgresql
# text query → CLIP text embedding → pgvector similarity search → return products

import os
import open_clip
import torch
import psycopg2
from dotenv import load_dotenv
load_dotenv()

url = os.getenv('URL')

device = "cuda" if torch.cuda.is_available() else "cpu"
# trade-off is that CLIP's text understanding is shallower
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.to(device)

def get_embedding(input):
    text_input = tokenizer([input]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(text_input)
    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze(0).cpu().numpy().tolist()
    
def find_products(query):
    embedding = get_embedding(query)
    
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    
    results = []
    try:
        cur.execute("""
            SELECT id, product_display_name, image_name, master_category, base_colour
            FROM products
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (embedding, ))
        results = cur.fetchall()
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()

    cur.close()
    conn.close()
    return results

query = "red hiking backpack with water bottle holder"
products = find_products(query)
print(f"Products: {products}")
