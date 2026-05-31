def memoize(fn):
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]

    return wrapper


@memoize
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
