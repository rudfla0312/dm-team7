"""Shared experiment helpers from the original Louvain and SVD runs."""
from collections import defaultdict
import csv
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_graph(diseases, similarities, k):
    graph = nx.Graph()
    graph.add_nodes_from(diseases)
    neighbors = {}
    for disease, row in zip(diseases, similarities):
        selected = row[:k]
        neighbors[disease] = selected
        for other, weight in selected:
            graph.add_edge(disease, other, weight=weight)
    return graph, neighbors


def feature_similarities(processed, diseases):
    index={d:i for i,d in enumerate(diseases)}
    edges=read_csv(processed/'disease_gene_t03.csv')
    if len(edges)!=810225: raise ValueError('Unexpected original feature data')
    # Match the CBF notebook's IDF universe: all diseases in the feature file.
    all_diseases={r['disease_id'] for r in edges}
    gene_disease=defaultdict(set)
    for r in edges: gene_disease[r['gene_id']].add(r['disease_id'])
    filtered=[r for r in edges if r['disease_id'] in index and float(r['evidence_score'])>=0.3]
    if len({(r['disease_id'],r['gene_id']) for r in filtered})!=len(filtered):
        raise ValueError('Duplicated disease-gene relationships')
    genes=sorted({r['gene_id'] for r in filtered}); gindex={g:i for i,g in enumerate(genes)}
    rr=np.array([index[r['disease_id']] for r in filtered],dtype=np.int32)
    cc=np.array([gindex[r['gene_id']] for r in filtered],dtype=np.int32)
    val=np.array([float(r['evidence_score']) for r in filtered],dtype=np.float64)
    x=coo_matrix((val,(rr,cc)),shape=(len(diseases),len(genes))).tocsr()
    idf=np.array([np.log((len(all_diseases)+1)/(len(gene_disease[g])+1))+1 for g in genes])
    sims={}
    for mode,mat in [('score_cosine',x),('idf_cosine',x.multiply(idf).tocsr())]:
        norm=np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
        inv=np.divide(1.0,norm,out=np.zeros_like(norm),where=norm>0)
        unit=diags(inv)@mat
        similarity=(unit@unit.T).toarray().astype(np.float32)
        np.fill_diagonal(similarity,0)
        sims[mode]=similarity
    return sims


def sorted_similarity_rows(diseases, matrix):
    ids=np.arange(len(diseases)); rows=[]
    for row in matrix:
        indices=ids[row>0]; values=row[indices]
        order=np.lexsort((indices,-values))
        rows.append([(diseases[int(i)],float(v)) for i,v in zip(indices[order],values[order])])
    return rows


def group_weights(sim, communities, diseases, power):
    group=np.empty(len(diseases),dtype=np.int32);lookup={d:i for i,d in enumerate(diseases)}
    for i,comm in enumerate(communities):
        for d in comm:group[lookup[d]]=i
    scores=np.where(group[:,None]==group[None,:],sim,0).astype(np.float32)
    if power==2: scores*=scores
    return csr_matrix(scores)


def communities_for(graph, resolution, seed):
    if graph.number_of_edges() == 0:
        return [{disease} for disease in graph.nodes]
    return nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)
