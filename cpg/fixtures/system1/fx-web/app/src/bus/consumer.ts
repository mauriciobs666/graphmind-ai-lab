import type { Consumer } from "kafkajs";
import { TOPIC_ORDERS, TOPIC_PREFIX } from "./topics";

export async function onOrderCreated(consumer: Consumer) {
  await consumer.subscribe({ topic: TOPIC_ORDERS });
}

export async function onOrderShipped(consumer: Consumer) {
  await consumer.subscribe({ topic: `${TOPIC_PREFIX}.shipped` });
}

export async function onDlq(consumer: Consumer) {
  await consumer.subscribe({ topic: "orders.created.dlq" });
}

export async function onConfigured(consumer: Consumer, config: { bus: { topic: string } }) {
  await consumer.subscribe({ topic: config.bus.topic });
}
