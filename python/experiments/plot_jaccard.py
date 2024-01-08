import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection", default="robust04")
parser.add_argument("-s" ,"--similarity", default="semantic")
args = parser.parse_args()

mean_jaccard_robust = [["CMP", 1, 0.00004],
                       ["CMP", 5, 0.006415],
                       ["CMP", 10, 0.225127],
                       ["CMP", 12.5, 0.511886],
                       ["CMP", 15, 0.772219],
                       ["CMP", 17.5, 0.914894],
                       ["CMP", 20, 0.965233],
                       ["CMP", 35, 0.987167],
                       ["CMP", 50, 0.987867],
                       ["Mhl", 1, 0.000091],
                       ["Mhl", 5, 0.004719],
                       ["Mhl", 10, 0.101067],
                       ["Mhl", 12.5, 0.258863],
                       ["Mhl", 15, 0.470178],
                       ["Mhl", 17.5, 0.679299],
                       ["Mhl", 20, 0.840926],
                       ["Mhl", 35, 0.986087],
                       ["Mhl", 50, 0.987667],
                       ["Vkr$_{CMP}$", 1, 0.00004],
                       ["Vkr$_{CMP}$", 5, 0.004656],
                       ["Vkr$_{CMP}$", 10, 0.095774],
                       ["Vkr$_{CMP}$", 12.5, 0.159416],
                       ["Vkr$_{CMP}$", 15, 0.187551],
                       ["Vkr$_{CMP}$", 17.5, 0.19642],
                       ["Vkr$_{CMP}$", 20, 0.195299],
                       ["Vkr$_{CMP}$", 35, 0.215867],
                       ["Vkr$_{CMP}$", 50, 0.238692],
                       ["Vkr$_{Mhl}$", 1, 0],
                       ["Vkr$_{Mhl}$", 5, 0.004713],
                       ["Vkr$_{Mhl}$", 10, 0.048803],
                       ["Vkr$_{Mhl}$", 12.5, 0.095945],
                       ["Vkr$_{Mhl}$", 15, 0.147183],
                       ["Vkr$_{Mhl}$", 17.5, 0.179329],
                       ["Vkr$_{Mhl}$", 20, 0.18635],
                       ["Vkr$_{Mhl}$", 35, 0.210636],
                       ["Vkr$_{Mhl}$", 50, 0.230734],
                       ["ArampatzisEtAl2011", 1, 0.200105],
                       ["ArampatzisEtAl2011", 50, 0.200105],
                       ["FroebeEtAl2021", 1, 0],
                       ["FroebeEtAl2021", 50, 0]]

mean_jaccard_msmarco = [["CMP", 1, 0],
                        ["CMP", 5, 0.002216],
                        ["CMP", 10, 0.109206],
                        ["CMP", 12.5, 0.298804],
                        ["CMP", 15, 0.537252],
                        ["CMP", 17.5, 0.730599],
                        ["CMP", 20, 0.855207],
                        ["CMP", 35, 0.973347],
                        ["CMP", 50, 0.976274],
                        ["Mhl", 1, 0],
                        ["Mhl", 5, 0.003893],
                        ["Mhl", 10, 0.051027],
                        ["Mhl", 12.5, 0.145285],
                        ["Mhl", 15, 0.291154],
                        ["Mhl", 17.5, 0.475256],
                        ["Mhl", 20, 0.647708],
                        ["Mhl", 35, 0.959818],
                        ["Mhl", 50, 0.975094],
                        ["Vkr$_{CMP}$", 1, 0],
                        ["Vkr$_{CMP}$", 5, 0.002312],
                        ["Vkr$_{CMP}$", 10, 0.05395],
                        ["Vkr$_{CMP}$", 12.5, 0.098655],
                        ["Vkr$_{CMP}$", 15, 0.139095],
                        ["Vkr$_{CMP}$", 17.5, 0.170792],
                        ["Vkr$_{CMP}$", 20, 0.166556],
                        ["Vkr$_{CMP}$", 35, 0.179868],
                        ["Vkr$_{CMP}$", 50, 0.20034],
                        ["Vkr$_{Mhl}$", 1, 0],
                        ["Vkr$_{Mhl}$", 5, 0.002082],
                        ["Vkr$_{Mhl}$", 10, 0.030045],
                        ["Vkr$_{Mhl}$", 12.5, 0.067521],
                        ["Vkr$_{Mhl}$", 15, 0.102638],
                        ["Vkr$_{Mhl}$", 17.5, 0.13498],
                        ["Vkr$_{Mhl}$", 20, 0.157314],
                        ["Vkr$_{Mhl}$", 35, 0.182259],
                        ["Vkr$_{Mhl}$", 50, 0.194064],
                        ["ArampatzisEtAl2011", 1, 0.33822],
                        ["ArampatzisEtAl2011", 50, 0.33822],
                        ["FroebeEtAl2021", 1, 0],
                        ["FroebeEtAl2021", 50, 0]]

semantic_msmarco = [["CMP", 1, 0.023809],
["CMP", 5, 0.031578],
["CMP", 10, 0.214002],
["CMP", 12.5, 0.458159],
["CMP", 15, 0.680512],
["CMP", 17.5, 0.82431],
["CMP", 20, 0.903154],
["CMP", 35, 0.950819],
["CMP", 50, 0.951883],
["Mhl", 1, 0.020172],
["Mhl", 5, 0.034226],
["Mhl", 10, 0.119329],
["Mhl", 12.5, 0.240695],
["Mhl", 15, 0.426888],
["Mhl", 17.5, 0.610283],
["Mhl", 20, 0.750299],
["Mhl", 35, 0.944804],
["Mhl", 50, 0.951499],
["Vkr$_{CMP}$", 1, 0.027715],
["Vkr$_{CMP}$", 5, 0.032322],
["Vkr$_{CMP}$", 10, 0.137075],
["Vkr$_{CMP}$", 12.5, 0.211305],
["Vkr$_{CMP}$", 15, 0.308401],
["Vkr$_{CMP}$", 17.5, 0.37223],
["Vkr$_{CMP}$", 20, 0.413493],
["Vkr$_{CMP}$", 35, 0.52183],
["Vkr$_{CMP}$", 50, 0.564539],
["Vkr$_{Mhl}$", 1, 0.022643],
["Vkr$_{Mhl}$", 5, 0.026084],
["Vkr$_{Mhl}$", 10, 0.083691],
["Vkr$_{Mhl}$", 12.5, 0.148717],
["Vkr$_{Mhl}$", 15, 0.215054],
["Vkr$_{Mhl}$", 17.5, 0.283593],
["Vkr$_{Mhl}$", 20, 0.333447],
["Vkr$_{Mhl}$", 35, 0.505195],
["Vkr$_{Mhl}$", 50, 0.553097],
["ArampatzisEtAl2011", 1, 0.50875],
["ArampatzisEtAl2011", 50, 0.50875],
["FroebeEtAl2021", 1, 0.076818],
["FroebeEtAl2021", 50, 0.076818]]


semantic_robust = [["CMP", 1, 0.073826],
["CMP", 5, 0.099685],
["CMP", 10, 0.395724],
["CMP", 12.5, 0.671666],
["CMP", 15, 0.870752],
["CMP", 17.5, 0.960824],
["CMP", 20, 0.987024],
["CMP", 35, 0.996128],
["CMP", 50, 0.996271],
["Mhl", 1, 0.077214],
["Mhl", 5, 0.095371],
["Mhl", 10, 0.244288],
["Mhl", 12.5, 0.426883],
["Mhl", 15, 0.626636],
["Mhl", 17.5, 0.79403],
["Mhl", 20, 0.906961],
["Mhl", 35, 0.995745],
["Mhl", 50, 0.996233],
["Vkr$_{CMP}$", 1, 0.076625],
["Vkr$_{CMP}$", 5, 0.100128],
["Vkr$_{CMP}$", 10, 0.278321],
["Vkr$_{CMP}$", 12.5, 0.411726],
["Vkr$_{CMP}$", 15, 0.510698],
["Vkr$_{CMP}$", 17.5, 0.577947],
["Vkr$_{CMP}$", 20, 0.622105],
["Vkr$_{CMP}$", 35, 0.731848],
["Vkr$_{CMP}$", 50, 0.760094],
["Vkr$_{Mhl}$", 1, 0.076335],
["Vkr$_{Mhl}$", 5, 0.096346],
["Vkr$_{Mhl}$", 10, 0.188144],
["Vkr$_{Mhl}$", 12.5, 0.281941],
["Vkr$_{Mhl}$", 15, 0.381768],
["Vkr$_{Mhl}$", 17.5, 0.472083],
["Vkr$_{Mhl}$", 20, 0.532779],
["Vkr$_{Mhl}$", 35, 0.705293],
["Vkr$_{Mhl}$", 50, 0.746436],
["ArampatzisEtAl2011", 1, 0.487301],
["ArampatzisEtAl2011", 50, 0.487301],
["FroebeEtAl2021", 1, 0.203499],
["FroebeEtAl2021", 50, 0.203499]]


mean_jaccard_robust = pd.DataFrame(mean_jaccard_robust, columns=["mechanism", "epsilon", "jaccard"])
mean_jaccard_msmarco = pd.DataFrame(mean_jaccard_msmarco, columns=["mechanism", "epsilon", "jaccard"])
semantic_robust = pd.DataFrame(semantic_robust, columns=["mechanism", "epsilon", "jaccard"])
semantic_msmarco = pd.DataFrame(semantic_msmarco, columns=["mechanism", "epsilon", "jaccard"])

c2d = {"jaccard": {'robust04': mean_jaccard_robust, 'msmarco-passage': mean_jaccard_msmarco}, "semantic": {'robust04': semantic_robust, 'msmarco-passage': semantic_msmarco}}

sns.set(font_scale=2.1, style="whitegrid")
plt.figure(figsize=(13.5, 9))

ax = sns.lineplot(data=c2d[args.similarity][args.collection], x="epsilon", y="jaccard", hue="mechanism", style="mechanism", markers=True)
plt.ylabel("similarity", fontsize=30)
plt.xlabel("$\epsilon$", fontsize=30)
plt.xticks([0, 5, 10, 15, 20, 35, 50])
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, frameon=False, fontsize=25)
ax.legend_.remove()
plt.savefig(f"../../data/figures/{args.similarity}-{args.collection}.pdf")


plt.show()
