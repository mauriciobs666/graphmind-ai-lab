package com.acme.fxlib.cli;

import com.acme.fxlib.Pricing;
import com.vendor.handles.Handle;
import java.lang.reflect.Method;
import java.math.BigDecimal;

/** Command-line front end for the pricing library. */
public class QuoteCli {

    private final Handle handle;
    private final int qty;
    private final String sku;

    public QuoteCli(Handle handle, int qty, String sku) {
        this.handle = handle;
        this.qty = qty;
        this.sku = sku;
    }

    public static void main(String[] args) throws Exception {
        String cfg = args.length > 0 ? args[0] : Pricing.class.getName();
        Method quote = Class.forName(cfg).getMethod("quote");
        System.out.println("pricing entry point: " + quote.getName());
    }

    public BigDecimal quoteDefault() {
        return Pricing.quote(1, "SKU-1", Pricing.DEFAULT_CURRENCY);
    }

    public String renderLegacy() {
        return String.valueOf(handle.quote(qty, sku));
    }
}
