import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, accuracy_score
import ast
import pickle

print("1. Loading dataset and building graph structure...")
df = pd.read_csv("jobs_clean_final_sbert_ready.csv") # <-- New CSV name

# We don't need ast.literal_eval anymore! We just split your clean string.
all_skills = set()
for skills_str in df['skills_text'].dropna():
    skills = [s.strip() for s in skills_str.split(';') if s.strip()]
    all_skills.update(skills)
    
skill_list = sorted(list(all_skills))
skill_to_id = {skill: i for i, skill in enumerate(skill_list)}

job_indices, skill_indices = [], []
for job_id, row in df.iterrows():
    if pd.isna(row['skills_text']): continue
    skills = [s.strip() for s in row['skills_text'].split(';') if s.strip()]
    for skill in skills:
        if skill in skill_to_id:
            job_indices.append(job_id)
            skill_indices.append(skill_to_id[skill])

print("2. Loading SBERT embeddings & creating graph...")
data = HeteroData()
data['job'].x = torch.load('job_embeddings.pt')
data['skill'].x = torch.load('skill_embeddings.pt')
data['job', 'requires', 'skill'].edge_index = torch.tensor([job_indices, skill_indices], dtype=torch.long)
data['skill', 'required_by', 'job'].edge_index = torch.tensor([skill_indices, job_indices], dtype=torch.long)

print("3. Defining and Training GNN...")
class BaseGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

base_model = BaseGNN(128, 64)
gnn_model = to_hetero(base_model, data.metadata(), aggr='sum')
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

def train():
    gnn_model.train()
    optimizer.zero_grad()
    out = gnn_model(data.x_dict, data.edge_index_dict)
    
    pos_edge_index = data['job', 'requires', 'skill'].edge_index
    pos_scores = (out['job'][pos_edge_index[0]] * out['skill'][pos_edge_index[1]]).sum(dim=1)
    
    neg_jobs = torch.randint(0, out['job'].size(0), (pos_edge_index.size(1),))
    neg_skills = torch.randint(0, out['skill'].size(0), (pos_edge_index.size(1),))
    neg_scores = (out['job'][neg_jobs] * out['skill'][neg_skills]).sum(dim=1)
    
    loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
           F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(150):
    loss = train()
    if epoch % 50 == 0 or epoch == 149:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")

print("\n4. Saving trained model and metadata...")
# Save the model weights
torch.save(gnn_model.state_dict(), 'gnn_weights.pth')
# Save the graph structure
torch.save(data, 'graph_data.pt')
# Save the dataset and vocab so our app can read them
with open('metadata.pkl', 'wb') as f:
    pickle.dump({'df': df, 'skill_list': skill_list, 'skill_to_id': skill_to_id}, f)

print("✅ Training complete and saved!")

# MODEL EVALUATION
print("\n🔍 Calculating Model Accuracy & AUC...")
gnn_model.eval()
with torch.no_grad():
    out = gnn_model(data.x_dict, data.edge_index_dict)
    
    # Get the real, actual connections
    pos_edge_index = data['job', 'requires', 'skill'].edge_index
    pos_scores = (out['job'][pos_edge_index[0]] * out['skill'][pos_edge_index[1]]).sum(dim=1)
    
    # Generate fake, random connections to test the model
    neg_jobs = torch.randint(0, out['job'].size(0), (pos_edge_index.size(1),))
    neg_skills = torch.randint(0, out['skill'].size(0), (pos_edge_index.size(1),))
    neg_scores = (out['job'][neg_jobs] * out['skill'][neg_skills]).sum(dim=1)
    
    # Combine everything for scoring
    y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu().numpy()
    y_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    
    # Calculate Accuracy (Using a 0.5 threshold)
    y_pred = (torch.sigmoid(torch.cat([pos_scores, neg_scores])) > 0.5).float().cpu().numpy()
    
    auc = roc_auc_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_pred)
    
    print("="*40)
    print("📊 FINAL MODEL PERFORMANCE")
    print("="*40)
    print(f"🎯 Binary Accuracy: {acc * 100:.2f}%")
    print(f"📈 AUC Score:       {auc * 100:.2f}%")
    print("="*40)