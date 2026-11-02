import { useEffect, useState } from "react";

export function Health() {
  const [ok, setOk] = useState(false);
  useEffect(() => {
    fetch("/api/health")
      .then((res) => setOk(res.ok))
      .catch(() => setOk(false));
  }, []);
  return <span>{ok ? "up" : "down"}</span>;
}
