import { useEffect, useState } from "react";

const JOB_BASE = import.meta.env.VITE_JOB_BASE;

export function JobPanel() {
  const [runs, setRuns] = useState(0);
  useEffect(() => {
    fetch(`${JOB_BASE}/job/status`)
      .then((res) => res.json())
      .then((body) => setRuns(body.runs));
  }, []);
  return <section>Runs: {runs}</section>;
}
