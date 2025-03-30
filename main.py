import streamlit as st
import os
import requests
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from Bio import PDB
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D

# Device setup
device = torch.device("cpu")

# Amino acid vocabulary
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
SEQ_LENGTH = 256
EMBEDDING_DIM = 32

# Function to download PDB files
def download_pdb(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"{pdb_id}.pdb", "w") as f:
            f.write(response.text)
        st.write(f"✅ {pdb_id}.pdb downloaded successfully.")
    else:
        raise FileNotFoundError(f"❌ PDB file {pdb_id} not found!")

# Manual three_to_one function
def three_to_one(resname):
    residue_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        'UNK': 'X'
    }
    return residue_map.get(resname, 'X')

# Extract amino acid sequence from PDB
def extract_sequence(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    seq = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                try:
                    seq += three_to_one(resname)
                except KeyError:
                    continue
    return seq[:SEQ_LENGTH].ljust(SEQ_LENGTH, "X")

# Encode sequence as integer indices
def encode_sequence(seq):
    return [AA_TO_IDX.get(aa, 0) for aa in seq]

# Compute contact map
def compute_contact_map(coords, threshold=8.0, fixed_size=256):
    num_residues = len(coords)
    contact_map = np.zeros((num_residues, num_residues))
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:
                contact_map[i, j] = 1
                contact_map[j, i] = 1
    img = Image.fromarray((contact_map * 255).astype(np.uint8)) # Convert to PIL Image
    img_resized = img.resize((fixed_size, fixed_size), Image.NEAREST)
    return np.array(img_resized) / 255.0

# Extract C-alpha coordinates
def extract_ca_coords(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
    return np.array(coords)

# Model with embedding layer
class ContactMapPredictor(nn.Module):
    def __init__(self, num_residues=256, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(len(AA_VOCAB), embedding_dim)
        self.fc1 = nn.Linear(num_residues * embedding_dim, 512)
        self.fc2 = nn.Linear(512, num_residues * num_residues)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

# Training function
def train(model, X_tensor, Y_tensor, epochs=20, lr=0.001):
    model.to(device)
    X_tensor, Y_tensor = X_tensor.to(device), Y_tensor.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            st.write(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Kabsch align function
def kabsch_align(P, Q):
    P_mean = np.mean(P, axis=0)
    Q_mean = np.mean(Q, axis=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean
    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    P_aligned = (P_centered @ R) + Q_mean
    return P_aligned

# Refine structure function
def refine_structure(coords, target_distances, lr=0.01, steps=1700):
    coords = torch.tensor(coords, requires_grad=True, dtype=torch.float32)
    target_distances = torch.tensor(target_distances, dtype=torch.float32)
    optimizer = optim.Adam([coords], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        dist_matrix = torch.cdist(coords, coords)
        loss = torch.mean((dist_matrix - target_distances) ** 2)
        loss.backward()
        optimizer.step()
    return coords.detach().numpy()

# Contact map to distance matrix function
def contact_map_to_distance_matrix(contact_map, min_dist=3.5, max_dist=20.0, scale=10):
    return min_dist + (max_dist - min_dist) * np.exp(-scale * contact_map)

# Compute RMSD function
def compute_rmsd(P, Q):
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))

# Streamlit app
def main():
    st.title("Protein 3D Structure Prediction")
    pdb_id = st.text_input("Enter PDB ID:", "1HHO")

    if st.button("Predict Structure"):
        try:
            download_pdb(pdb_id)
            seq = extract_sequence(f"{pdb_id}.pdb")
            coords = extract_ca_coords(f"{pdb_id}.pdb")
            contact_map = compute_contact_map(coords)
            X_tensor = torch.tensor([encode_sequence(seq)], dtype=torch.long).to(device)
            Y_tensor = torch.tensor([contact_map.flatten()], dtype=torch.float32).to(device)

            model = ContactMapPredictor().to(device)
            st.write(f"Training on {pdb_id}")
            train(model, X_tensor, Y_tensor)

            with torch.no_grad():
                pred_contact_map = model(X_tensor).cpu().numpy().reshape(256, 256)

            st.subheader("Predicted Contact Map")
            fig_contact, ax_contact = plt.subplots(figsize=(5, 5))
            ax_contact.imshow(pred_contact_map, cmap="gray_r")
            st.pyplot(fig_contact)

            distance_matrix = contact_map_to_distance_matrix(pred_contact_map)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)

            iso = Isomap(n_components=3, n_neighbors=10)
            coords_3d = iso.fit_transform(distance_matrix)

            real_coords = extract_ca_coords(f"{pdb_id}.pdb")

            coords_3d = refine_structure(coords_3d, distance_matrix)

            aligned_coords = kabsch_align(coords_3d, real_coords[:len(coords_3d)])

            st.subheader("Real vs Predicted 3D Structure")
            fig_3d = plt.figure(figsize=(10, 7))
            ax_3d = fig_3d.add_subplot(111, projection="3d")
            ax_3d.scatter(aligned_coords[:, 0], aligned_coords[:, 1], aligned_coords[:, 2], c="b", marker="o", label="Predicted")
            ax_3d.scatter(real_coords[:, 0], real_coords[:, 1], real_coords[:, 2], c="r", marker="o", label="Real")
            ax_3d.set_xlabel("X-axis")
            ax_3d.set_ylabel("Y-axis")
            ax_3d.set_zlabel("Z-axis")
            ax_3d.legend()
            st.pyplot(fig_3d)

            rmsd = compute_rmsd(aligned_coords, real_coords[:len(coords_3d)])
            dmax = np.max(np.linalg.norm(real_coords - real_coords[:, None], axis=-1))
            relative_accuracy = max(0, 100 * (1 - (rmsd / dmax)))
            st.write(f"Relative Accuracy: {relative_accuracy:.2f}%")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
