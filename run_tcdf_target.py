import torch
import pandas as pd
import argparse
from TCDF import findcauses
import networkx as nx
import matplotlib.pyplot as plt
import pylab

# === Parse CLI arguments ===
parser = argparse.ArgumentParser(description='Run TCDF for a given dataset and target')
parser.add_argument('--file', type=str, required=True, help='CSV file with time series data')
parser.add_argument('--target', type=str, required=True, help='Target column (WL or SLA)')
parser.add_argument('--cuda', action="store_true", default=False, help='Use GPU if available')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--kernel_size', type=int, default=4)
parser.add_argument('--hidden_layers', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--log_interval', type=int, default=500)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dilation_c', type=int, default=4)
parser.add_argument('--significance', type=float, default=0.8)
args = parser.parse_args()

# === Parameters ===
file = args.file
target = args.target
cuda = args.cuda
epochs = args.epochs
kernel_size = args.kernel_size
levels = args.hidden_layers + 1
lr = args.learning_rate
optimizername = args.optimizer
log_interval = args.log_interval
seed = args.seed
dilation_c = args.dilation_c
significance = args.significance

# === Run TCDF ===
df = pd.read_csv(file)
if target not in df.columns:
    raise ValueError(f"Target '{target}' not found in dataset.")

print(f"\n📊 Running TCDF on {file} with target: {target}")
causes, delays, loss, scores = findcauses(
    target=target,
    cuda=cuda,
    epochs=epochs,
    kernel_size=kernel_size,
    layers=levels,
    log_interval=log_interval,
    lr=lr,
    optimizername=optimizername,
    seed=seed,
    dilation_c=dilation_c,
    significance=significance,
    file=file
)

# === Plot causal graph ===
columns = list(df.columns)
G = nx.DiGraph()
for col in columns:
    G.add_node(col)
for (effect_idx, cause_idx), delay in delays.items():
    G.add_edge(columns[cause_idx], columns[effect_idx], weight=delay)

pos = nx.circular_layout(G)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_color='white', edgecolors='black', node_size=1000)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title(f"Causal Graph for target: {target}")
pylab.show()

# === Output summary ===
print("\n✅ Validated causal links and delays:")
for (effect_idx, cause_idx), delay in delays.items():
    print(f"{columns[cause_idx]} ➜ {columns[effect_idx]}  (delay: {delay})")
