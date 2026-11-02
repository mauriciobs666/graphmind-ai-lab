package com.acme.fxlib.bus;

import java.math.BigDecimal;

/** A price that was quoted for an item. */
public record PriceEvent(String sku, BigDecimal total) {
}
