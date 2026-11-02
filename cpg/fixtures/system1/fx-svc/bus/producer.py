"""Order event producer."""
import os

from kafka import KafkaProducer

from fx_svc.bus.constants import PREFIX, TOPIC_ORDERS

producer = KafkaProducer()


class _Alerts:
    def created(self, o):
        return f"alert {o.id}"


_HANDLERS = _Alerts()


def publish_alert(o, evt):
    handler = getattr(_HANDLERS, evt)
    return handler(o)


def publish_dynamic(o):
    producer.send(os.environ["ORDERS_TOPIC"], o)


def publish_created(o):
    producer.send(TOPIC_ORDERS, o)


def publish_shipped(o):
    producer.send(f"{PREFIX}.shipped", o)
