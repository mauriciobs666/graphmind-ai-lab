package com.acme.fxlib.bus;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/** Marks a method as the handler for events published under a topic. */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface FxSubscribe {
    String value();
}
