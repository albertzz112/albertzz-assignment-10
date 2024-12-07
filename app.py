from flask import Flask, request, render_template
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
import os

# Initialize Flask app
app = Flask(__name__)

# Configuration
DATASET_DIR = "static/coco_images_resized"
EMBEDDINGS_FILE = "image_embeddings.pickle"
model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = get_tokenizer('ViT-B-32')
model.eval()
embeddings_df = pd.read_pickle(EMBEDDINGS_FILE)

# Utility function for similarity calculation
def calculate_similarities(query_embedding, top_k=5):
    similarities = [
        F.cosine_similarity(query_embedding, torch.tensor(row['embedding']).unsqueeze(0)).item()
        for _, row in embeddings_df.iterrows()
    ]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    return [(embeddings_df.iloc[i]['file_name'], similarities[i]) for i in top_indices]

@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query_type = request.form.get('query_type')
        top_k = 5
        results = []

        if query_type == 'text':
            text_query = request.form['text_query']
            tokenized_text = tokenizer([text_query])
            query_embedding = F.normalize(model.encode_text(tokenized_text), p=2, dim=-1)
            results = calculate_similarities(query_embedding, top_k)
        
        elif query_type == 'image':
            image_file = request.files['image_query']
            image = Image.open(image_file)
            processed_image = preprocess(image).unsqueeze(0)
            query_embedding = F.normalize(model.encode_image(processed_image), p=2, dim=-1)
            results = calculate_similarities(query_embedding, top_k)
        
        elif query_type == 'hybrid':
            image_file = request.files['image_query']
            text_query = request.form['text_query']
            weight = float(request.form['weight'])

            image = Image.open(image_file)
            processed_image = preprocess(image).unsqueeze(0)
            image_embedding = F.normalize(model.encode_image(processed_image), p=2, dim=-1)

            tokenized_text = tokenizer([text_query])
            text_embedding = F.normalize(model.encode_text(tokenized_text), p=2, dim=-1)

            hybrid_embedding = F.normalize(weight * text_embedding + (1 - weight) * image_embedding, p=2, dim=-1)
            results = calculate_similarities(hybrid_embedding, top_k)
        
        return render_template('results.html', results=results, dataset_dir=DATASET_DIR)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
