package com.acme.fxlib.bus;

import java.util.logging.Logger;

/** Records every quoted price. */
public class AuditListener {

    private static final Logger LOG = Logger.getLogger(AuditListener.class.getName());

    public static void install() {
        FxBus.register(new AuditListener());
    }

    @FxSubscribe("price.updated")
    public void onPrice(PriceEvent event) {
        LOG.info("quoted " + event.sku() + " at " + event.total());
    }
}
