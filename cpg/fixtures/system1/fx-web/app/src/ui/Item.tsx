import { useEffect, useState } from "react";

export function ItemView({ id }: { id: number }) {
  const [name, setName] = useState("");
  useEffect(() => {
    fetch(`/api/items/${id}`)
      .then((res) => res.json())
      .then((body) => setName(body.name));
  }, [id]);
  return <h2>{name}</h2>;
}
