"""Order graph in the graph store."""


def link_order(graph):
    return graph.query("CREATE (:Order {id: 1})")


def neighbors(graph):
    return graph.query("MATCH (o:Order) RETURN o.id")
