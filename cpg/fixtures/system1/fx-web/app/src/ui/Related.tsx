import { useEffect, useState } from "react";
import { API_PREFIX } from "../api/endpoints";

const RELATED_ID = 7;

export function Related() {
  const [name, setName] = useState("");
  useEffect(() => {
    fetch(`${API_PREFIX}/items/${RELATED_ID}`)
      .then((res) => res.json())
      .then((body) => setName(body.name));
  }, []);
  return <li>{name}</li>;
}
