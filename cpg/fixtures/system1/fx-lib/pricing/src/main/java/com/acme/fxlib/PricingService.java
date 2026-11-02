package com.acme.fxlib;

import com.acme.fxlib.proto.PricingGrpc;
import com.acme.fxlib.proto.QuoteReply;
import com.acme.fxlib.proto.QuoteRequest;
import io.grpc.stub.StreamObserver;
import java.util.Currency;
import java.util.Locale;

/** Server side of the pricing service declared in pricing.proto. */
public class PricingService extends PricingGrpc.PricingImplBase {

    @Override
    public void quote(QuoteRequest request, StreamObserver<QuoteReply> responseObserver) {
        Currency currency = Currency.getInstance(request.getCurrency());
        String total = Pricing.quote(request.getQuantity(), request.getSku(), currency, Locale.ROOT);
        responseObserver.onNext(QuoteReply.newBuilder().setTotal(total).build());
        responseObserver.onCompleted();
    }
}
