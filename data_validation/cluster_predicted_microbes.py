import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from adjustText import adjust_text
import matplotlib as mpl
import matplotlib.font_manager as fm
import os
from scipy.cluster.hierarchy import dendrogram, linkage

font_path = "/user_data/yezy/zhangjm/.font/Times New Roman.ttf"

        font_prop = fm.FontProperties(weight='bold', style='italic', size=14)
        font_prop_normal = fm.FontProperties(size=12)

features = np.load("bacteria_wang_features.npy")
with open("bacteria_wang_names.txt", "r", encoding="utf-8") as f:
    names = [line.strip() for line in f]

best_k = 2
best_score = -1
for k in range(2, 8):
   
    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = clustering.fit_predict(features)
    
    if len(set(labels)) > 1:
        score = silhouette_score(features, labels)
        print(f"k={k}, Silhouette Score={score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k
    else:
        print(f"k={k}")

print(f"best_k: k={best_k}")

clustering = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
labels = clustering.fit_predict(features)

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

cluster_members = {}
for i in range(best_k):
    member_indices = np.where(labels == i)[0]
    member_names = [names[idx] for idx in member_indices]
    cluster_members[i] = member_names
    
    print(f"\nCluster {i+1} ({len(member_names)} members):")
    for name in member_names:
        print(f"  - {name}")

plt.figure(figsize=(18, 8))
plt.title('Hierarchical Clustering Dendrogram', fontsize=18, fontproperties=font_prop)
plt.xlabel('Sample Index', fontsize=14, fontproperties=font_prop_normal)
plt.ylabel('Distance', fontsize=14, fontproperties=font_prop_normal)

Z = linkage(features, 'ward')

plt.figure(figsize=(14, 12))  
colors = plt.cm.Set1(np.linspace(0, 1, best_k))


scatter_handles = []  
for i in range(best_k):
    idx = labels == i
    scatter = plt.scatter(
        reduced[idx, 0], 
        reduced[idx, 1], 
        c=[colors[i]], 
        label=f"Cluster {i+1} ({len(cluster_members[i])} members)", 
        s=120,  
        edgecolor='k',
        alpha=0.8
    )
    scatter_handles.append(scatter)

texts = []
for i, (x, y) in enumerate(reduced):

    display_name = names[i] if len(names[i]) <= 15 else names[i][:12] + "..."
    text = plt.text(
        x, y, 
        display_name, 
        fontsize=20, 
        fontweight='bold',  
        fontstyle='italic',  
        fontproperties=font_prop,  
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')  
    )
    texts.append(text)

try:
    adjust_text(
        texts, 
        arrowprops=dict(arrowstyle='-', color='gray', lw=1.0),  
        expand_points=(1.3, 1.7),  
        expand_text=(1.2, 1.3),
        force_text=0.7,
        only_move={'points':'y', 'text':'xy'},
        va='center', 
        ha='center',
        precision=0.01  
    )
except NameError:
    offset = 0.03 * (reduced.max() - reduced.min())  
    for text in texts:
        pos = text.get_position()
        text.set_position((pos[0] + offset * (1 if np.random.rand() > 0.5 else -1), 
                          pos[1] + offset * (1 if np.random.rand() > 0.5 else -1)))

plt.title(
    f"Hierarchical Clustering Results (k={best_k})", 
    fontsize=22,  
    fontweight='bold', 
    fontproperties=font_prop
)
plt.xlabel(
    "Principal Component 1", 
    fontsize=22,  
    fontweight='bold', 
    fontproperties=font_prop_normal
)
plt.ylabel(
    "Principal Component 2", 
    fontsize=22, 
    fontweight='bold', 
    fontproperties=font_prop_normal
)

legend = plt.legend(
    handles=scatter_handles,  
    loc='best', 
    framealpha=0.7, 
    prop=font_prop,  
    title='Clusters',
    title_fontproperties=font_prop,  
    markerscale=1.5  
)

plt.grid(True, linestyle='--', alpha=0.3)
plt.gca().set_axisbelow(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tick_params(axis='both', which='major', labelsize=14)

plt.gca().set_facecolor('#f8f8f8')

plt.tight_layout()
plt.savefig("microbe_hierarchical_cluster.svg", bbox_inches='tight')
plt.savefig("microbe_hierarchical_cluster.png", dpi=300, bbox_inches='tight')
plt.show()

