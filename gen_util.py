from random import randint, shuffle

def rand_perm(m, n):
    ps = [i for i in range(m, n+1)]
    shuffle(ps)
    return ps

def gen_tree(n):
    es = list()
    for i in range(2, n+1):
        es.append((randint(1, i-1), i))
    return es

def primes(n):
    ps = list()
    b = [True]*n
    for p in range(2, n):
        if p * p > n: break
        if b[p]:
            for q in range(p * p, n, p): b[q] = False
    for p in range(2, n):
        if b[p]:
            ps.append(p)
    return ps

def gen_conn_graph(n, m):
    es = set()
    for i in range(2, n+1):
        es.add((randint(1, i-1), i))
    while len(es) < m:
        a, b = randint(1, n), randint(1, n)
        if a < b:
            es.add((a, b))
    return es
