package com.acme.fxlib.bus;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;
import java.util.logging.Logger;

/** A minimal in-process publish/subscribe bus. */
public final class FxBus {

    private static final Logger LOG = Logger.getLogger(FxBus.class.getName());

    private static final Map<String, List<Consumer<Object>>> HANDLERS = new ConcurrentHashMap<>();

    private FxBus() {
    }

    public static void subscribe(String topic, Consumer<Object> handler) {
        HANDLERS.computeIfAbsent(topic, key -> new CopyOnWriteArrayList<>()).add(handler);
    }

    public static void publish(String topic, Object event) {
        for (Consumer<Object> handler : HANDLERS.getOrDefault(topic, List.of())) {
            handler.accept(event);
        }
    }

    /** Subscribes every {@link FxSubscribe} method of the listener. */
    public static void register(Object listener) {
        for (Method method : listener.getClass().getDeclaredMethods()) {
            FxSubscribe marker = method.getAnnotation(FxSubscribe.class);
            if (marker == null) {
                continue;
            }
            subscribe(marker.value(), event -> {
                try {
                    method.invoke(listener, event);
                } catch (ReflectiveOperationException e) {
                    LOG.warning("listener failed: " + e);
                }
            });
        }
    }

    /** Subscribes a logging handler to the topic named by the launch configuration. */
    public static void subscribeConfigured(String listenerId) {
        String topic = System.getProperty("bus.topic");
        if (topic == null) {
            return;
        }
        subscribe(topic, event -> LOG.info(listenerId + " received " + event));
    }
}
