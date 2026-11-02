import { useEffect, useState } from "react";

export function JobStatus() {
  const [state, setState] = useState("unknown");
  useEffect(() => {
    fetch("http://localhost:8000/job/status")
      .then((res) => res.json())
      .then((body) => setState(body.state));
  }, []);
  return <span>{state}</span>;
}
