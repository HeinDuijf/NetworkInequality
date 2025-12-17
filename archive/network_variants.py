# def randomize_network(G, n_edges: int):
#     # Check if the graph is directed
#     is_directed = G.is_directed()

#     # Get edges and nodes
#     edges = copy.deepcopy(list(G.edges()))
#     random.shuffle(edges)
#     edges_set = set(edges)
#     new_edges_set = copy.deepcopy(edges_set)
#     nodes = copy.deepcopy(list(G.nodes()))

#     # Find which edges to remove
#     to_remove_set = set(random.sample(edges, k=n_edges))
#     new_edges_set.difference_update(to_remove_set)

#     # Generate a new edges
#     for edge in to_remove_set:
#         new_edge = (random.choice(nodes), random.choice(nodes))
#         if not is_directed:
#             new_edge = tuple(sorted(new_edge))  # Ensure (u, v) == (v, u) for undirected graphs

#         # Avoid duplicate edges and self-loops
#         while (new_edge in new_edges_set) or (new_edge[0] == new_edge[1]):
#             new_edge = (random.choice(nodes), random.choice(nodes))
#             if not is_directed:
#                 new_edge = tuple(sorted(new_edge))

#         new_edges_set.add(new_edge)

#     # Create a new graph with updated edges
#     G_new = copy.deepcopy(G)
#     G_new.remove_edges_from(to_remove_set)
#     G_new.add_edges_from(new_edges_set)

#     return G_new


# ## Equalize
def equalize(net: nx.DiGraph, n: int) -> nx.DiGraph:
    """
    Equalize the network by rewiring n random edges.
    """
    equalized_net = copy.deepcopy(net)
    triangles = get_triangles(net)
    rewired_triangles = random.sample(triangles, n)

    for triangle in rewired_triangles:
        edge = triangle[-2:]  # Take the last two nodes as the edge to be rewired
        # Remove edge
        # I: What is the difference between the two conditions?
        if equalized_net.has_edge(*edge):
            equalized_net.remove_edge(*edge)
        elif equalized_net.has_edge(edge[1], edge[0]):
            equalized_net.remove_edge(edge[1], edge[0])
        else:
            continue

        # Add new edge to create a new triangle that passes by the first node
        node = triangle[0]
        neighbors = list(net.predecessors(node)) + list(net.successors(node))
        # I: I understand k=10 neighbors so that there are enough options to choose from,
        sources_sample = random.choices(neighbors, k=20)
        targets_sample = random.choices(neighbors, k=20)
        edge_sample = [
            (source, target)
            for source in sources_sample
            for target in targets_sample
            if source != target and not equalized_net.has_edge(source, target)
        ]
        new_edge = random.choice(
            edge_sample
        )  # Throws an error if no edges are available
        equalized_net.add_edge(*new_edge)
    return equalized_net


# # ## Densify
# def densify_fancy_speed_up(
#     net: nx.DiGraph, n_edges: int, target_degree_dist: str = "original",
#     target_average_clustering: float = None,
#     keep_density_fixed = False,
# ) -> nx.DiGraph:
#     """
#     Densifies a directed network by adding new edges to increase its density,
#     while optionally targeting a specific degree distribution and clustering coefficient.
#     Priority is given to targeting the specified clustering coefficient.

#     Parameters
#     ----------
#     net : nx.DiGraph
#         The original directed network to densify.
#     n_edges : int
#         The number of edges to add.
#     target_degree_dist : str, optional
#         The target degree distribution for new edges.
#         "original" preserves the original degree distribution,
#         "uniform" assigns equal probability to all nodes. Default is "original".
#     target_clustering : float, optional
#         The desired average clustering coefficient. If None, uses the original network's clustering.

#     Returns
#     -------
#     nx.DiGraph
#         A new directed network with increased density and optionally modified clustering/degree distribution.
#     """

#     # Create a copy of the original network
#     net_new = copy.deepcopy(net)

#     if target_average_clustering is None:
#         target_average_clustering = nx.average_clustering(net)
#     if target_degree_dist == "original":
#         # Use the original degree distribution
#         out_degrees = dict(net.out_degree())
#         in_degrees = dict(net.in_degree())
#     if target_degree_dist == "uniform":
#         out_degrees = {node: 1 for node in net.nodes()}
#         in_degrees = {node: 1 for node in net.nodes()}

#     if keep_density_fixed:
#         edges_to_remove = random.sample(net_new.edges(), n_edges)
#         net_new.remove_edges_from(edges_to_remove)

#     clustering_dict: dict = nx.clustering(net_new)

#     # Add edges in neighborhoods
#     n_edges_added = 0
#     edges_added_clustering = 0
#     edges_added_degree_dist = 0
#     new_average_clustering = np.average(list(clustering_dict.values()))
#     while n_edges_added < n_edges:
#         if new_average_clustering < target_average_clustering:
#             # Add new edge to increase clustering
#             node = random.choice(list(net.nodes()))
#             neighbors = list(net.predecessors(node)) + list(net.successors(node))
#             out_degrees_neighbors = {node: out_degrees[node] for node in neighbors}
#             in_degrees_neighbors = {node: in_degrees[node] for node in neighbors}
#             out_weights = out_degrees_neighbors.values()
#             if all(out_weights) == 0:
#                 out_weights = np.ones(len(out_degrees_neighbors.keys()))
#             in_weights = in_degrees_neighbors.values()

#             if all(in_weights) == 0:
#                 in_weights = np.ones(len(in_degrees_neighbors.keys()))

#             sources = random.choices(list(out_degrees_neighbors.keys()), weights=out_weights, k=10)
#             targets = random.choices(list(in_degrees_neighbors.keys()), weights=in_weights, k=10)
#             possible_edges = [
#                 (source, target) for source in sources for target in targets
#                 if source != target and not net_new.in_edges(source, target)
#             ]
#             if possible_edges != []:
#                 new_edge = random.choice(possible_edges)
#                 n_edges_added += 1
#                 net_new.add_edge(*new_edge)
#                 neighborhood_0 = list(net_new.predecessors(new_edge[0])) + list(net_new.successors(new_edge[0]))
#                 neighborhood_1 = list(net_new.predecessors(new_edge[1])) + list(net_new.successors(new_edge[1]))
#                 affected_nodes = [new_edge[0], new_edge[1]] + list(set(neighborhood_0).intersection(set(neighborhood_1)))
#                 for node in affected_nodes:
#                     clustering_dict[node] = nx.clustering(net_new, node)
#                 new_average_clustering = np.average(list(clustering_dict.values()))
#                 edges_added_clustering += 1
#         else:
#             # Add new edge based on target degree distribution
#             sources_sample = random.choices(list(out_degrees.keys()), weights=out_degrees.values(), k=10)
#             targets_sample = random.choices(list(in_degrees.keys()), weights=in_degrees.values(), k=10)
#             edge_sample = [
#                 (source, target)
#                 for source in sources_sample
#                 for target in targets_sample
#                 if source != target and not net_new.has_edge(source, target)]
#             if edge_sample != []:
#                 new_edge = random.choice(edge_sample) # Throws an error if no edges are available
#                 n_edges_added += 1
#                 net_new.add_edge(*new_edge)
#                 neighborhood_0 = list(net_new.predecessors(new_edge[0])) + list(net_new.successors(new_edge[0]))
#                 neighborhood_1 = list(net_new.predecessors(new_edge[1])) + list(net_new.successors(new_edge[1]))
#                 affected_nodes = [new_edge[0], new_edge[1]] + list(set(neighborhood_0).intersection(set(neighborhood_1)))
#                 for node in affected_nodes:
#                     clustering_dict[node] = nx.clustering(net_new, node)
#                 new_average_clustering = np.average(list(clustering_dict.values()))
#                 edges_added_degree_dist += 1
#         # print(f"{n_edges_added=:,} edges added")
#     # print(f"{edges_added_clustering:,} edges added to increase clustering")
#     # print(f"{edges_added_degree_dist:,} edges added based on {target_degree_dist} degree distribution")
#     return net_new


# ## Cluster
def decluster(net: nx.DiGraph, n_triangles: int) -> nx.DiGraph:
    """
    Decluster the network by rewiring n_triangles random triangles.
    """
    decluster_net = copy.deepcopy(net)
    triangles = get_triangles(net)
    rewired_triangles = random.sample(triangles, n_triangles)
    rewired_edges = [
        (source, target) for (source, target, _) in rewired_triangles
    ]  # Warning: triangles are based on undirected graph!

    for edge in rewired_edges:
        # Remove edge
        if decluster_net.has_edge(*edge):
            decluster_net.remove_edge(*edge)
        elif decluster_net.has_edge(edge[1], edge[0]):
            decluster_net.remove_edge(edge[1], edge[0])
        else:
            continue
        # I: I like this but maybe the new edge generates a new cluster?
        # Add new edge based on out- and in-degree distribution
        out_degrees = dict(net.out_degree())
        in_degrees = dict(net.in_degree())
        sources_sample = random.choices(
            list(out_degrees.keys()), weights=out_degrees.values(), k=10
        )
        targets_sample = random.choices(
            list(in_degrees.keys()), weights=in_degrees.values(), k=10
        )
        edge_sample = [
            (source, target)
            for source in sources_sample
            for target in targets_sample
            if source != target and not decluster_net.has_edge(source, target)
        ]
        new_edge = random.choice(
            edge_sample
        )  # Throws an error if no edges are available
        decluster_net.add_edge(*new_edge)
    return decluster_net


def cluster_network(net: nx.DiGraph, n: int) -> nx.DiGraph:
    # Create a copy of the original network
    cluster_net = copy.deepcopy(net)

    # Add edges based on the degree distribution
    n_edges_to_add = n
    # print(f"{n_edges_to_add=:,}")

    # Add edges in neighborhoods
    edges_new = []
    # I: wouldn't it be better to add one edge per random chosen node?
    # I: this way we can ensure that the new edges are not making a single node too 'cliqued'
    while len(edges_new) < n_edges_to_add:
        node = random.choice(list(net.nodes()))
        neighbors = list(net.predecessors(node)) + list(net.successors(node))
        out_degrees_neighbors = dict(net.out_degree(neighbors))
        in_degrees_neighbors = dict(net.in_degree(neighbors))
        out_weights = out_degrees_neighbors.values()
        if all(out_weights) == 0:
            out_weights = np.ones(len(out_degrees_neighbors.keys()))
        in_weights = in_degrees_neighbors.values()

        if all(in_weights) == 0:
            in_weights = np.ones(len(in_degrees_neighbors.keys()))

        sources = random.choices(
            list(out_degrees_neighbors.keys()), weights=out_weights, k=10
        )
        targets = random.choices(
            list(in_degrees_neighbors.keys()), weights=in_weights, k=10
        )
        possible_edges = [
            (source, target)
            for source in sources
            for target in targets
            if source != target
            and not (source, target) in edges_new
            and not net.in_edges(source, target)
        ]
        if possible_edges != []:
            edges_new.append(random.choice(possible_edges))
    cluster_net.add_edges_from(edges_new)

    return cluster_net


# def densify_fancy_speed_up_v2(
#     net,
#     n_edges,
#     target_degree_dist="original",
#     target_average_clustering=None,
#     keep_density_fixed=False
# ):
#     """
#     Densifies a directed network by adding (or replacing) edges to increase its density,
#     optionally targeting a specific degree distribution and clustering coefficient.
#     If keep_density_fixed is True, the total number of edges is preserved.

#     Parameters
#     ----------
#     net : nx.DiGraph
#         The original directed network to densify.
#     n_edges : int
#         The number of edges to add (or replace if keep_density_fixed is True).
#     target_degree_dist : str
#         "original" uses the original degree distribution; "uniform" gives equal weight to all nodes.
#     target_average_clustering : float or None
#         Target average clustering coefficient. If None, uses current average.
#     keep_density_fixed : bool
#         If True, removes one edge per new edge to keep edge count constant.

#     Returns
#     -------
#     nx.DiGraph
#         The modified graph.
#     """
#     net_new = copy.deepcopy(net)

#     if target_average_clustering is None:
#         target_average_clustering = nx.average_clustering(net)

#     if target_degree_dist == "original":
#         out_degrees = dict(net.out_degree())
#         in_degrees = dict(net.in_degree())
#     elif target_degree_dist == "uniform":
#         out_degrees = {node: 1 for node in net.nodes()}
#         in_degrees = {node: 1 for node in net.nodes()}
#     else:
#         raise ValueError("target_degree_dist must be 'original' or 'uniform'")

#     clustering_dict = nx.clustering(net_new)

#     n_edges_added = 0
#     new_average_clustering = sum(clustering_dict.values()) / len(clustering_dict)
#     max_attempts = n_edges * 10
#     attempts = 0

#     while n_edges_added < n_edges and attempts < max_attempts:
#         attempts += 1

#         if keep_density_fixed and len(net_new.edges) > 0:
#             old_edge = random.choice(list(net_new.edges()))
#             net_new.remove_edge(*old_edge)

#         if new_average_clustering < target_average_clustering:
#             node = random.choice(list(net.nodes()))
#             neighbors = list(net.predecessors(node)) + list(net.successors(node))
#             if not neighbors:
#                 continue

#             out_neighbors = {n: out_degrees.get(n, 1) for n in neighbors}
#             in_neighbors = {n: in_degrees.get(n, 1) for n in neighbors}

#             sources = random.choices(list(out_neighbors.keys()), weights=out_neighbors.values(), k=10)
#             targets = random.choices(list(in_neighbors.keys()), weights=in_neighbors.values(), k=10)

#             possible_edges = [
#                 (s, t) for s in sources for t in targets
#                 if s != t and not net_new.has_edge(s, t)
#             ]

#             if possible_edges:
#                 new_edge = random.choice(possible_edges)
#                 net_new.add_edge(*new_edge)
#                 n_edges_added += 1

#                 neighborhood_0 = list(net_new.predecessors(new_edge[0])) + list(net_new.successors(new_edge[0]))
#                 neighborhood_1 = list(net_new.predecessors(new_edge[1])) + list(net_new.successors(new_edge[1]))
#                 affected = set([new_edge[0], new_edge[1]]) | (set(neighborhood_0) & set(neighborhood_1))

#                 for n in affected:
#                     clustering_dict[n] = nx.clustering(net_new, n)
#                 new_average_clustering = sum(clustering_dict.values()) / len(clustering_dict)
#         else:
#             sources = random.choices(list(out_degrees.keys()), weights=out_degrees.values(), k=10)
#             targets = random.choices(list(in_degrees.keys()), weights=in_degrees.values(), k=10)

#             possible_edges = [
#                 (s, t) for s in sources for t in targets
#                 if s != t and not net_new.has_edge(s, t)
#             ]

#             if possible_edges:
#                 new_edge = random.choice(possible_edges)
#                 net_new.add_edge(*new_edge)
#                 n_edges_added += 1

#                 neighborhood_0 = list(net_new.predecessors(new_edge[0])) + list(net_new.successors(new_edge[0]))
#                 neighborhood_1 = list(net_new.predecessors(new_edge[1])) + list(net_new.successors(new_edge[1]))
#                 affected = set([new_edge[0], new_edge[1]]) | (set(neighborhood_0) & set(neighborhood_1))

#                 for n in affected:
#                     clustering_dict[n] = nx.clustering(net_new, n)
#                 new_average_clustering = sum(clustering_dict.values()) / len(clustering_dict)

#     return net_new
#     return net_new
#     return net_new
#     return net_new
