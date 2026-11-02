"""Report views over the table the nightly job fills."""
from fastapi import APIRouter

from fx_svc.db.orders import connect

router = APIRouter()


@router.get("/reports/{report_id}")
def report_view(report_id: int):
    conn = connect()
    rows = conn.execute("SELECT * FROM order_report").fetchall()
    return {"id": report_id, "rows": rows[:50]}
