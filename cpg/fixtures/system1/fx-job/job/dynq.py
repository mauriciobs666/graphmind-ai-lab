"""Ad-hoc table dump used by operators."""


def dump_table(conn, cols, tbl):
    return conn.execute("SELECT " + cols + " FROM " + tbl).fetchall()
