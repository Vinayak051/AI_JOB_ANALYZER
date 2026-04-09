import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

print("1. Loading the newly cleaned dataset...")
df = pd.read_csv("jobs_clean_final_sbert_ready.csv")

print("2. Loading SBERT Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("3. Generating Job Embeddings from the new 'sbert_text'...")
# SBERT will now read your perfectly formatted "Title | Skills | Body" text
job_texts = df['sbert_text'].fillna("").tolist()
job_embeddings = model.encode(job_texts, convert_to_tensor=True)

print("4. Extracting the clean skill vocabulary...")
# We rebuild our skill list from your 'skills_text' column
all_skills = set()
for skills_str in df['skills_text'].dropna():
    # Your script separated skills with ' ; '
    skills = [s.strip() for s in skills_str.split(';') if s.strip()]
    all_skills.update(skills)

skill_list = sorted(list(all_skills))

print(f"🎯 Filtered down to {len(skill_list)} highly relevant skills!")
print("5. Generating Skill Embeddings...")
skill_embeddings = model.encode(skill_list, convert_to_tensor=True)

print("6. Saving new tensors for the GNN...")
torch.save(job_embeddings, 'job_embeddings.pt')
torch.save(skill_embeddings, 'skill_embeddings.pt')

print("✅ Boom! New embeddings generated and saved.")