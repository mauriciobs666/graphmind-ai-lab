"""gRPC client for the pricing service."""


def quote(channel, request):
    call = channel.unary_unary("/acme.fxlib.Pricing/Quote")
    return call(request)


def quote_dynamic(stub, method_name, request):
    return getattr(stub, method_name)(request)
