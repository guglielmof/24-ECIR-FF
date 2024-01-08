import os
import ir_datasets
from sentence_transformers import SentenceTransformer, util, InputExample
import faiss
import numpy as np
from tqdm import tqdm
from time import time
from torch.nn.parallel import DataParallel
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset


def preprocess_document(doc):
    title = doc.title.lower().replace("\n", " ")
    body = doc.body.lower().replace("\n", " ")
    text = (title + " " + body)  # namedtuple<doc_id, title, body, marked_up_doc>

    return doc.doc_id, text


# Load the model
transformer = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')


class CustomModel(torch.nn.Module):
    # Our model

    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model

    def forward(self, input):
        start = time()
        output = self.model.encode(input, convert_to_tensor=True)
        print(f"batch done encoded: {len(output)} in {time() - start:.2f} s")
        return output


model = CustomModel(transformer)

model = DataParallel(model)

os.environ['JAVA_HOME'] = '/ssd/data/faggioli/SOFTWARE/jdk-11.0.11'

dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")

start = time()
k = 0
data = []
for d in dataset.docs_iter():
    data.append(preprocess_document(d)[1])
    k += 1
    if k == 10000:
        break


class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.data = sentences
        self.len = len(sentences)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


batch_size = 500
data_loader = DataLoader(dataset=CustomDataset(data), batch_size=batch_size, shuffle=True)

# Convert sentences to embeddings using the model

docs_emb = []
with torch.no_grad():
    for batch in data_loader:
        input = batch
        output = model(input)
        docs_emb.append(output)

docs_emb = torch.cat(docs_emb, dim=0)
print(f"done in {time() - start:.2f} s")

faiss_index = faiss.IndexFlatL2(768)  # L2 distance (Euclidean distance) index for 128-dimensional vectors

faiss_index.add(docs_emb)

faiss.write_index(faiss_index, "/ssd/data/faggioli/24-ECIR-FF/data/indexes/tasb")


query = "does china accept political compromises?"

query_vector = transformer.encode(query)

k = 10
distances, indices = faiss_index.search(np.array([query_vector]), k)

print("Distances:")
print(distances)
print(indices)
