import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
import PyPDF2
import re
import pickle
import pandas as pd

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="AI Job Matcher", page_icon="🎯", layout="centered")

st.title("🎯 AI-Powered Job Matcher & Skill Gap Analyzer")
st.markdown("""
This engine uses a **Graph Neural Network (GNN)** to map your skills against the structural reality of the job market. 
Upload your resume or type your skills below to discover your top matches and what you need to learn next.
""")

# ==========================================
# 2. LOAD MODEL (CACHED)
# ==========================================
class BaseGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

@st.cache_resource
def load_system():
    with open('metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
    df, skill_list, skill_to_id = meta['df'], meta['skill_list'], meta['skill_to_id']

    data = torch.load('graph_data.pt', weights_only=False)
    base_model = BaseGNN(128, 64)
    gnn_model = to_hetero(base_model, data.metadata(), aggr='sum')
    gnn_model.load_state_dict(torch.load('gnn_weights.pth', weights_only=True))
    gnn_model.eval()
    
    return df, skill_list, skill_to_id, data, gnn_model

df, skill_list, skill_to_id, data, gnn_model = load_system()

# ==========================================
# 3. THE LOGIC
# ==========================================
def extract_skills_from_pdf(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + " "
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    
    found_skills = []
    for skill in skill_list:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text):
            found_skills.append(skill)
    return found_skills

def get_recommendations(user_skills_input, top_n=5):
    with torch.no_grad():
        out = gnn_model(data.x_dict, data.edge_index_dict)
        final_job_embeddings, final_skill_embeddings = out['job'], out['skill']
    
    # CRITICAL FIX: Aggressively strip whitespace from the user input
    user_skills_lower = [s.strip().lower() for s in user_skills_input]
    user_skill_ids = [skill_to_id[s] for s in user_skills_lower if s in skill_to_id]
    
    # Tell the UI exactly if/why the skills weren't found
    if not user_skill_ids:
        st.error(f"None of these skills ({', '.join(user_skills_lower)}) were found in the trained graph vocabulary. The model can't route them!")
        return None
        
    user_skill_vectors = final_skill_embeddings[user_skill_ids]
    user_profile_vector = torch.mean(user_skill_vectors, dim=0).unsqueeze(0)
    
    similarities = F.cosine_similarity(user_profile_vector, final_job_embeddings)
    top_scores, top_indices = torch.topk(similarities, top_n)
    
    results = []
    for score, idx in zip(top_scores, top_indices):
        job_row = df.iloc[idx.item()]
        match_percent = (score.item() + 1) / 2 * 100  
        
        # Safely parse the semicolon-separated string from the new dataset
        raw_skills = str(job_row.get('skills_text', ""))
        if pd.isna(raw_skills) or raw_skills.lower() == "nan":
            required_skills = []
        else:
            required_skills = [s.strip() for s in raw_skills.split(';') if s.strip()]
            
        missing_skills = [s for s in required_skills if s.lower() not in user_skills_lower]
        title = job_row.get('title_clean_std', job_row.get('title_clean', 'Unknown Title'))
        
        results.append({
            'title': title.title(),
            'score': match_percent,
            'required': required_skills,
            'missing': missing_skills
        })
    return results

# ==========================================
# 4. THE USER INTERFACE
# ==========================================
st.divider()

# Interactive Tabs
tab1, tab2 = st.tabs(["📄 Upload Resume", "⌨️ Type Skills"])

user_skills = []
analyze_triggered = False

with tab1:
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Analyze Resume", use_container_width=True):
            with st.spinner("Extracting skills..."):
                user_skills = extract_skills_from_pdf(uploaded_file)
            if user_skills:
                st.success(f"Extracted {len(user_skills)} valid skills: {', '.join(user_skills[:8])}...")
                analyze_triggered = True
            else:
                st.warning("Could not extract any matching technical skills from this PDF.")

with tab2:
    manual_input = st.text_input("Enter your skills (separated by commas)", placeholder="e.g., python, react, sql, machine learning")
    if st.button("Find Matches", use_container_width=True):
        if manual_input:
            user_skills = [s.strip().lower() for s in manual_input.split(',')]
            analyze_triggered = True
        else:
            st.warning("Please enter at least one skill.")

# ==========================================
# 5. RENDER THE RESULTS
# ==========================================
if analyze_triggered and user_skills:
    st.divider()
    st.subheader("🚀 Top Job Matches & Actionable Skill Gaps")
    
    with st.spinner("Traversing the Graph Neural Network..."):
        recommendations = get_recommendations(user_skills)
        
    if recommendations:
        for i, rec in enumerate(recommendations):
            # Render a beautiful UI card for each job
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### {i+1}. {rec['title']}")
                    st.markdown(f"**Required Skills:** {', '.join(rec['required'][:10])}")
                    
                    if len(rec['missing']) == 0:
                        st.success("🔥 Perfect Technical Match! You have no skill gaps for this role.")
                    else:
                        st.info(f"💡 **To reach 95%+ match, learn:** {', '.join(rec['missing'][:4])}")
                
                with col2:
                    st.metric(label="Match Score", value=f"{rec['score']:.1f}%")
                    st.progress(int(rec['score']))