import { useState } from "react";

export function OrderTotal() {
  const [total, setTotal] = useState<number | null>(null);
  const place = async () => {
    const res = await fetch("/api/v1/orders", { method: "POST" });
    const body = await res.json();
    setTotal(body.total);
  };
  return (
    <div>
      <button onClick={place}>Place order</button>
      <span>{total === null ? "-" : total}</span>
    </div>
  );
}
