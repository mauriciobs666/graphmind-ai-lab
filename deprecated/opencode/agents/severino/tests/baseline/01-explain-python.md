# 01-explain-python

- **Agent:** severino
- **Model:** <agent default>
- **Generated:** 2026-05-31T18:58:44-03:00
- **Duration:** 22s
- **Attached:** `memoize.py`

## Prompt

```
Explain what the attached `memoize.py` does, step by step. Cover how the closure and cache work, and what kinds of functions it is safe to use on.
```

## Response

**What `memoize.py` does – step‑by‑step**

| Line | Code | What happens |
|------|------|--------------|
| 1 | `def memoize(fn):` | A *decorator* named **memoize** is defined. It takes one argument, `fn`, which must be a callable (i.e., a function). |
| 2 | `cache = {}` | Inside the decorator we create an empty dictionary called **cache**. This will store previously‑computed results keyed by their arguments (`args`). |
| 4‑9 | `def wrapper(*args):` <br> `if args not in cache:` <br> `    cache[args] = fn(*args)` <br> `return cache[args]` | The **wrapper** is the function that will actually be applied to any function you decorate. It receives *any* number of positional arguments (`*args`).  <br> - If those arguments have not been seen before, it calls the original function `fn(*args)`, stores the result in `cache[args]`. <br> - In either case (already cached or newly computed) it returns that stored value. |
| 10 | `return wrapper` | The decorator returns the **wrapper** function, which means that when you write `@memoize`, Python will replace the decorated function with this wrapper. |
| 12‑16 | Example usage of `@memoize` on a naïve recursive Fibonacci implementation (`fib`). | Because each call to `fib(n)` is recomputed from scratch, the naive version has exponential time complexity. The decorator makes it *linear* by caching results for each integer `n`. |

---

### How the closure and cache work together

1. **Closure** – When we create `memoize(fn)`, the inner function `wrapper` “closes over” (i.e., captures) the variable `cache` defined in its outer scope. This means every call to a memoized function will have access to the same dictionary, preserving results across calls.

2. **Cache as a lookup table** – The key is the tuple of arguments (`args`). Because Python uses hashable objects for dict keys, only immutable types (e.g., integers, strings, tuples of those) can be stored. If `args` are mutable (lists, sets), they cannot be used directly; you’d need to convert them to a hashable representation first.

3. **Thread‑safety** – The current implementation is *not* thread‑safe. If two threads call the same function at once, one could read/modify the cache while the other is writing it, leading to race conditions or lost updates.

---

### Safe functions you can memoize

| ✅ Good | ❌ Problematic |
|--------|----------------|
| **Pure functions** that only depend on their arguments and return a single value (e.g., `fib`, arithmetic ops). | Functions with side‑effects that write to global state outside the cache. |
| Functions whose arguments are hashable (ints, strings, tuples of such). | Functions that receive mutable objects as args (lists) – they’ll be treated as distinct keys each time. |
| Recursive or expensive functions where repeated calls happen with same inputs (e.g., `pow`, `math.factorial`). | Functions that allocate large data structures inside the cache (memory blow‑up). |

**In practice:** Use `@memoize` on any pure, fast‑to‑compute function whose arguments are hashable. Avoid it for functions that already share state or need per‑thread isolation; you could wrap the call in a `threading.Lock()` manually if needed.

---

### Bottom line

- The file implements a simple **decorator** (`memoize`) that caches results keyed by their arguments.
- It turns an exponential algorithm (like naïve recursion) into a linear one by remembering past computations.
- Works safely for pure, hashable‑argument functions; otherwise it may cause bugs or memory issues.

