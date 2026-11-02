"""Generic helpers whose query text is built at runtime."""


def count_rows(conn, table):
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()


def match_label(graph, label):
    return graph.query("MATCH (n:%s) RETURN n" % label)
