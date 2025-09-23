# Requires: rdflib, networkx (optional)
# pip install rdflib networkx

from rdflib import Graph, URIRef, BNode, Literal
from rdflib import Namespace
from rdflib.namespace import RDF, RDFS
from typing import Set, Iterable

def extract_n_hop_rdflib(g: Graph,
                         seed_iri: str,
                         n: int,
                         direction: str = "both",
                         include_literals: bool = True) -> Graph:
    """
    Extract n-hop subgraph around seed_iri from an rdflib.Graph `g`.
    direction: "out", "in", or "both"
    include_literals: whether to add literal objects into the result (True by default)
    Returns an rdflib.Graph containing all triples of the subgraph.
    """
    seed = URIRef(seed_iri)
    result = Graph()
    visited_nodes: Set = set()
    frontier: Set = {seed}

    for hop in range(1, n+1):
        next_frontier: Set = set()
        if not frontier:
            break
        for node in frontier:
            # outgoing triples: node as subject
            if direction in ("out", "both"):
                for s, p, o in g.triples((node, None, None)):
                    # optionally skip literal objects from being considered as hops
                    result.add((s, p, o))
                    if isinstance(o, (URIRef, BNode)) and (o not in visited_nodes) and (o not in frontier):
                        next_frontier.add(o)

            # incoming triples: node as object
            if direction in ("in", "both"):
                for s, p, o in g.triples((None, None, node)):
                    result.add((s, p, o))
                    if isinstance(s, (URIRef, BNode)) and (s not in visited_nodes) and (s not in frontier):
                        next_frontier.add(s)

        # mark frontier as visited and move on
        visited_nodes.update(frontier)
        frontier = next_frontier

    # optionally prune dangling literal-only nodes if include_literals=False
    if not include_literals:
        to_remove = []
        for s, p, o in result:
            if isinstance(o, Literal):
                to_remove.append((s, p, o))
        for t in to_remove:
            result.remove(t)

    return result

# Example usage:
if __name__ == "__main__":
    g = Graph()
    g.parse("datasets/sciqa/project_data/orkg-dump-14-02-2023.nt", format="nt")

    seed = "http://orkg.org/orkg/resource/R150599" # CHEMDNER corpus
    sub = extract_n_hop_rdflib(g, seed, n=1, direction="both")
        # Define your namespace
    ORKGP = Namespace("http://orkg.org/orkg/predicate/")
    # Bind it to the graph
    sub.bind("orkgp", ORKGP)
    # Now serialize will use the bound prefix
    sub.serialize("chemdnerCorpus_1hop.ttl", format="turtle")
    print("triples:", len(sub))
    # save
    sub.serialize("chemdnerCorpus_1hop.ttl", format="turtle")
