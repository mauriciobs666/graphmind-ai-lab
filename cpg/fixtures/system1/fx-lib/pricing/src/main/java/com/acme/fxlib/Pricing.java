package com.acme.fxlib;

import com.acme.fxlib.bus.FxBus;
import com.acme.fxlib.bus.PriceEvent;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.NumberFormat;
import java.util.Currency;
import java.util.Locale;
import java.util.Map;

/** Price quotes for catalogue items. */
public final class Pricing {

    public static final Currency DEFAULT_CURRENCY = Currency.getInstance("USD");

    private static final Map<String, BigDecimal> UNIT_PRICES = Map.of(
            "SKU-1", new BigDecimal("9.99"),
            "SKU-2", new BigDecimal("24.50"));

    private Pricing() {
    }

    /** Quotes a total in the given currency and announces it on the bus. */
    public static BigDecimal quote(int qty, String sku, Currency currency) {
        BigDecimal total = amount(qty, sku, currency);
        FxBus.publish("price.updated", new PriceEvent(sku, total));
        return total;
    }

    /** Quotes a total and formats it for a locale. */
    public static String quote(int qty, String sku, Currency currency, Locale locale) {
        NumberFormat format = NumberFormat.getCurrencyInstance(locale);
        format.setCurrency(currency);
        return format.format(amount(qty, sku, currency));
    }

    private static BigDecimal amount(int qty, String sku, Currency currency) {
        BigDecimal unit = UNIT_PRICES.getOrDefault(sku, BigDecimal.ZERO);
        return unit.multiply(BigDecimal.valueOf(qty))
                .setScale(currency.getDefaultFractionDigits(), RoundingMode.HALF_EVEN);
    }
}
