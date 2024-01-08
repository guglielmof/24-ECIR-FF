import numpy as np
import pandas as pd
from glob import glob
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection")
parser.add_argument("-m", "--measure", default="nDCG@10")
args = parser.parse_args()

files = glob(f"../../data/measures/{args.collection}*.csv")


def parse_file(f):
    idx = f.split("/")[-1].replace("_TF_IDF_", "_TFIDF_").split("_")
    if len(idx) == 5:
        collection, model, mechanism, epsilon, embeddings = idx
    else:
        collection, model, mechanism, _ = idx
        epsilon = np.NAN
        embeddings = None

    df = pd.read_csv(f, sep="\t")
    df['collection'] = collection
    df['model'] = model
    df['mechanism'] = mechanism
    df['epsilon'] = float(epsilon)
    df['embeddings'] = embeddings

    return df


df = pd.concat([parse_file(f) for f in files])

map = {"MahalanobisMechanism": "Mhl", "VickreyCMPMechanism": "Vkr$_{CMP}$", "VickreyMMechanism": "Vkr$_{M}$", "CMPMechanism": "CMP",
       "original": "original"}
df.mechanism = df.mechanism.apply(lambda x: map[x])

df_orig = df.loc[(df.mechanism == "original") & (df.measure == args.measure)][['query_id', 'model', 'value']]
df_pert = df.loc[(df.mechanism != "original") & (df.measure == args.measure)][['query_id', 'epsilon', 'model', "mechanism", 'value']] \
    .merge(df_orig, on=["query_id", "model"], suffixes=("", "_orig"))

df_pert['loss'] = (df_pert['value_orig'] - df_pert['value']) / df_pert['value_orig']
df_pert['loss'] = df_pert.loss.fillna(0)

# Define the custom sorting order
custom_order = ["BM25", "TF-IDF", "glove", "tasb", "contriever"]
# Use the Categorical data type with custom order for sorting
df_pert['model'] = pd.Categorical(df_pert['model'], categories=custom_order, ordered=True)

# Define the custom sorting order
custom_order = ["CMP", "Mhl", "Vkr$_{CMP}$", "Vkr$_{M}$", ]
# Use the Categorical data type with custom order for sorting
df_pert['mechanism'] = pd.Categorical(df_pert['mechanism'], categories=custom_order, ordered=True)
df_pert.sort_values(by=["model", "mechanism"], inplace=True)

sns.set(font_scale=2, style="whitegrid")
graph = sns.FacetGrid(df_pert, col="model", hue="mechanism")
graph.map(sns.lineplot, "epsilon", "value")
graph.fig.set_size_inches(20, 7)
for i in range(len(df_pert.model.unique())):
    graph.facet_axis(0, i).set_title(graph.facet_axis(0, i).get_title().replace("model = ", ""))
graph.set_ylabels(args.measure)
# graph = sns.lineplot(data=df_pert, x="epsilon", y="value", hue="mechanism")
graph.add_legend()

sns.move_legend(
    graph, "lower center",
    bbox_to_anchor=(.5, 0.1), ncol=4, title=None, frameon=False,
)
plt.xticks([0, 5, 12, 20, 50])
plt.savefig(f"../../data/figures/plot_performance-{args.collection}.pdf")

perfs = df_pert[['epsilon', 'model', "mechanism", 'value']].groupby(
    ["epsilon", "mechanism", "model"]).mean().reset_index()

pd.options.display.float_format = '{:.3f}'.format
for mechanism in perfs.mechanism.unique():
    print(mechanism)
    tmp = perfs[perfs.mechanism == mechanism].pivot_table(index="model", columns="epsilon", values="value")
    print(tmp.to_string())

drop = df_pert[df_pert.loss != -np.inf][['epsilon', 'model', "mechanism", 'loss']].groupby(
    ["epsilon", "mechanism", "model"]).mean().reset_index()
for mechanism in drop.mechanism.unique():
    print(mechanism)
    tmp = drop[drop.mechanism == mechanism].pivot_table(index="model", columns="epsilon", values="loss")
    print(tmp.to_string())
