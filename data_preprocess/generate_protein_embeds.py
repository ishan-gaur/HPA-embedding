import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
from tape import ProteinBertModel, TAPETokenizer


protein_df = pd.read_csv(
    "/mnt/hdd1/users/aktgpt/HPA-embedding/stats/protein_seqs.csv", index_col=0
)
model = ProteinBertModel.from_pretrained("bert-base")
tokenizer = TAPETokenizer(
    vocab="iupac"
)  # iupac is the vocab for TAPE models, use unirep for the UniRep model

save_path = "/mnt/hdd2/datasets/hpa_data/protein_embeddings/tape_bert_base"

for i, row in tqdm(protein_df.iterrows(), total=protein_df.shape[0]):
    ensembl_ids = list(row["Ensembl_id"].split(","))
    for ensembl_id in ensembl_ids:
        sequence = row["Sequence"]
        token_ids = torch.tensor([tokenizer.encode(sequence)])
        output = model(token_ids)
        sequence_output = output[0].cpu().detach().numpy()
        np.save(f"{save_path}/{ensembl_id}.npy", sequence_output)
