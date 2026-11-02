import { useEffect, useState } from "react";

const REPORTS_BASE = import.meta.env.VITE_REPORTS_BASE;
const REPORT_ID = 3;

export function Reports() {
  const [title, setTitle] = useState("");
  useEffect(() => {
    fetch(`${REPORTS_BASE}/api/reports/${REPORT_ID}`)
      .then((res) => res.json())
      .then((body) => setTitle(body.title));
  }, []);
  return <h3>{title}</h3>;
}
