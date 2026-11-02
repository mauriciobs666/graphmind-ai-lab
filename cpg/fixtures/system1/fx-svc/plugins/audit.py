"""Audit plugin."""


class AuditPlugin:
    def run(self, evt):
        return ("audit", evt)
