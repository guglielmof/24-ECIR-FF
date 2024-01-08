import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("../../data/manual/radar_plot_data.csv", sep="\t")

def radar_plot(df, hue=None, yticks=[0.3, 0.7], fill=True):
    if hue is not None:
        variables = [c for c in df.columns if c != hue]
    else:
        variables = df.columns

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(len(variables)) * 2 * math.pi for n in range(len(variables))]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], variables, color='grey', size=50)

    # Draw ylabels
    ax.set_rlabel_position(0)

    plt.yticks(yticks, yticks, color="grey", size=30)
    plt.ylim(0, 0.95)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    if hue is None:
        values = list(df.iloc[0]) + [df.iloc[0][0]]
        print(angles, values)
        # Plot data
        ax.plot(angles, values, colors[0], linewidth=1, linestyle='solid')

        # Fill area
        if fill:
            ax.fill(angles, values, colors[0], alpha=0.5)

    else:
        for e, l in enumerate(df[hue].unique()):
            values = list(df.loc[df[hue]==l, variables].iloc[0])
            values += [values[0]]
            ax.plot(angles, values, colors[e], linewidth=3, linestyle='solid', label=l)

            # Fill area
            if fill:
                ax.fill(angles, values, colors[e], alpha=0.05)
        return ax
#plt.legend()

data['obf'] = 1-data['similarity']

data = data[~(data["obfuscation"]=="Vkr$_{Mhl}$ $\epsilon$=17.5")]

k = 0
for c in ["robust04", "msmarco-passage"]:
    for m in ["BM25", "Contriever"]:
        plt.figure(figsize=(24, 15))
        #radar_plot(data[(data["obfuscation"]=="Vkr$_{Mhl}$ ($\epsilon$=10)") & (data["collection"]=="msmarco-passage") & (data["model"]=="BM25")][['obf', 'nDCG@10', 'recall']])
        ax = radar_plot(data[(data["collection"]==c) & (data["model"]==m)][['obfuscation', 'obf', 'nDCG@10', 'recall']], hue="obfuscation", fill=False)
        if k==0:
            ax.legend(loc='center left', bbox_to_anchor=(-0.6, 0.5), ncols=1, frameon=False, fontsize=45)
            k = 1
        plt.savefig(f"../../data/figures/radar_plot_{c}_{m}.pdf")

        plt.show()
