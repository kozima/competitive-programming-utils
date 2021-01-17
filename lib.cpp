#include <limits>
#include <stack>
#include <functional>
#include <tuple>
#include <map>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <deque>
#include <chrono>
#include <queue>
#include <cassert>
#include <random>
#include <set>
#include <vector>
#include <iostream>
#include <cmath>

cin.tie(0); ios::sync_with_stdio(false);

long long lcm(long long a, long long b) {
    long long g = a, h = b;
    while (h > 0) { long long t = g; g = h; h = t % h; }
    return a / g * b;
}

long long gcd(long long a, long long b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    if (a < b) swap(a, b);
    while (b > 0) { long long t = a; a = b; b = t % b; }
    return a;
}

int extgcd(int a, int b, int& x, int &y) {
    if (b == 0) { x = 1; y = 0; return a; }
    else {
        int g = extgcd(b, a%b, y, x);
        y -= a / b * x;
        return g;
    }
}

long long modexp(int x, long long e, int m) {
    long long ans = 1, p = x % m;
    while (e > 0) {
        if (e % 2 != 0) ans = (ans * p) % m;
        p = (p * p) % m;
        e >>= 1;
    }
    return ans;
}

// int modinv(int a, int m) {
//     int x, y;
//     extgcd(a, m, x, y);
//     return x;
// }

// int modinv(int a, int m) { return modexp(a, m-2, m); }

long long modinv(int a, int m) {
    int x = m, y = a, p = 1, q = 0, r = 0, s = 1;
    while (y != 0) {
        int u = x / y;
        int x0 = y; y = x - y * u; x = x0;
        int r0 = p - r * u, s0 = q - s * u;
        p = r; r = r0; q = s; s = s0;
    }
    return q < 0 ? q + m : q;
}

// CRT
// return z < lcm(a, b) s.t. z = x mod a and z = y mod b
// z = -1 if there is no solution
// assume lcm(a, b) is not too large

long long modinv(long long a, long long m) {
    long long x = m, y = a, p = 1, q = 0, r = 0, s = 1;
    while (y != 0) {
        long long u = x / y;
        long long x0 = y; y = x - y * u; x = x0;
        long long r0 = p - r * u, s0 = q - s * u;
        p = r; r = r0; q = s; s = s0;
    }
    return q < 0 ? q + m : q;
}

// solve: z = x (mod a), z = y (mod b)
// a, b should fit into int32
long long CRT(long long a, long long x, long long b, long long y) {
    long long g = a, h = b;
    while (h > 0) { long long t = g; g = h; h = t % h; }
    if ((y - x) % g != 0) return -1;
    long long t = (y - x) / g % b * modinv(a/g, b/g) % b;
    long long ans = (x + a * t) % (a/g*b);
    if (ans < 0) ans += a/g*b;
    return ans;
}


// floor(a/b)
long long floor(long long a, long long b) {
    if (b < 0) { a = -a; b = -b; }
    return (a >= 0) ? a/b : (a+1)/b - 1;
}

// ceil(a/b)
long long ceil(long long a, long long b) {
    if (b < 0) { a = -a; b = -b; }
    return (a > 0) ? (a-1)/b + 1 : a/b;
}

// modular factorial----------------
long long fact[100002], facti[100002];

// initialization
fact[0] = 1; facti[0] = 1;
for (int i = 1; i <= n+1; i++) {
    fact[i] = (fact[i-1] * i) % M;
    facti[i] = modinv(fact[i], M);
}

// modular combination
/* usage:
    ModComb mc(n, M);
    ...
    long long comb = mc.get(p, q); // p <= n
*/

class ModComb {
    long long *fact, *facti;
    const int mod;
public:
    explicit ModComb(int n, int m) : mod(m) {
        fact = new long long[n+1];
        facti = new long long[n+1];
        fact[0] = 1; facti[0] = 1;
        for (int i = 1; i <= n; i++) fact[i] = (fact[i-1] * i) % m;
        // calc 1/n!
        long long &inv = facti[n], pw = fact[n];
        inv = 1;
        for (int e = mod-2; e > 0; e /= 2) {
            if (e&1) inv = inv * pw % mod;
            pw = pw * pw % mod;
        }
        for (int i = n-1; i > 0; i--) facti[i] = (facti[i+1] * (i+1)) % m;
    }

    ~ModComb() {
        if (fact) delete[] fact;
        if (facti) delete[] facti;
    }

    long long getFact(int n) const {
        return fact[n];
    }

    long long getFactInv(int n) const {
        return facti[n];
    }

    long long getComb(int n, int k) const {
        if (n < 0 || k < 0 || k > n) return 0;
        return fact[n] * facti[k] % mod * facti[n-k] % mod;
    }

    long long getPerm(int n, int k) const {
        if (n < 0 || k < 0 || k > n) return 0;
        return fact[n] * facti[n-k] % mod;
    }
};


// euler's totient
vector<int> totient(int n) {
    vector<int> res(n+1);
    vector<bool> b(n+1, true);
    for (int i = 0; i <= n; i++) res[i] = i;
    for (int p = 2; p * p <= n; p++)
        if (b[p]) for (int q = p * p; q <= n; q += p) b[q] = false;
    for (int p = 2; p <= n; p++)
        if (b[p]) for (int q = p; q <= n; q += p) res[q] = res[q] / p * (p - 1);
    return res;
}


// BIT (binary indexed tree)
int bit[N+1] = {};
long long ans = 0;
for (int i = 0; i <= n; i++) {
    // sum below r[i] (inclusive)
    for (int j = r[i]; j > 0; j -= j&-j) ans += bit[j];
    // increment r[i]
    for (int j = r[i]; j <= n+1; j += j&-j) bit[j]++;
}


// random shuffle (Fisher-Yates)
template<typename T>
void random_shuffle(vector<T> &a) {
    // different output every time
    // mt19937 mt{ random_device{}() };

    // always the same output
    mt19937 mt;

    for (int i = a.size() - 1; i >= 1; i--) {
        uniform_int_distribution<int> dist(0, i-1);
        int j = dist(mt);
        swap(a[i], a[j]);
    }
}

int main() {
    // different output every time
    // mt19937 mt{ random_device{}() };

    // always get the same output
    mt19937 mt;

    uniform_int_distribution<long long> dist(1, (long long)1e18);

    vector<int> v(20);
    for (int i = 0; i < 20; i++) v[i] = i;
    for ( int i = 0 ; i != 1000 ; ++i ) {
        random_shuffle(v);
        for (int x : v) cout << x << ' '; cout << endl;
    }
}


// uf/union-find

class UnionFind {
    vector<int> par, rank;
public:
    explicit UnionFind(int n) : par(n), rank(n, 0) {
        iota(par.begin(), par.end(), 0);
    }

    int find(int i) {
        return (par[i] == i) ? i : (par[i] = find(par[i]));
    }

    void unite(int i, int j) {
        int x = find(i), y = find(j);
        if (x != y) {
            if (rank[x] < rank[y])
                par[x] = y;
            else {
                par[y] = x;
                if (rank[x] == rank[y]) rank[x]++;
            }
        }
    }

    bool same(int i, int j) { return find(i) == find(j); }
};


// undo-able union find with weight of each component
// without path compression
class UnionFind {
    int *par, *rank;
    long long *weight;
    struct Undo_info {
        int x, y, rx, ry;
        long long wx, wy;
    };
    map<pair<int, int>, Undo_info> undo_info;
public:
    UnionFind(int n, int *x) {
        par = new int[n];
        iota(par, par+n, 0);
        rank = new int[n];
        fill(rank, rank+n, 1);
        weight = new long long[n];
        copy(x, x+n, weight);
    };
    int find(int i) {
        while (par[i] != i) i = par[i];
        return i;
    }
    int same(int i, int j) { return find(i) == find(j); }
    void unite(int i, int j) {
        int x = find(i), y = find(j);
        if (x == y) return;
        undo_info[make_pair(i, j)] = {x, y, rank[x], rank[y], weight[x], weight[y]};
        if (rank[x] < rank[y]) {
            par[x] = y;
            weight[y] += weight[x];
        } else {
            par[y] = x;
            if (rank[x] == rank[y]) rank[x]++;
            weight[x] += weight[y];
        }
    }
    void undo(int i, int j) {
        assert (undo_info.find(make_pair(i, j)) != undo_info.end());
        Undo_info info = undo_info[make_pair(i, j)];
        int x = info.x, y = info.y;
        par[x] = x;
        par[y] = y;
        rank[x] = info.rx;
        rank[y] = info.ry;
        weight[x] = info.wx;
        weight[y] = info.wy;
        undo_info.erase(make_pair(i, j));
    }
    long long get_weight(int i) { return weight[find(i)]; }
};


// cycle detection
// example: set dist[i] = 0 for all i in cycle
int dist[3001];
vector<int> adj[3001];
bool vis[3001];

int dfs(int i, int p) {
    vis[i] = true;
    for (int j : adj[i]) {
        if (j == p) continue;
        if (vis[j]) {
            dist[i] = 0;        // i in cycle
            return j;
        }
        int v = dfs(j, i);
        if (v == -1) // found a cycle not including i
            return v;
        if (v >= 0) { // found a cycle including i and ending with v
            dist[i] = 0;        // i in cycle
            return v == i ? -1 : v;
        }
    }
    return -2; // no cycle
}


// SA template
// https://gist.github.com/kozima/8055794ff5be3f2bf15048f0ff2e6153

unsigned int xor128() {
    static unsigned int x=123456789, y=362436069, z=521288629, w=88675123;
    unsigned int t;
    t=(x^(x<<11)); x=y; y=z; z=w;
    return (w=(w^(w>>19))^(t^(t>>8)));
}

inline bool rand_bool(double prob) {
    constexpr double x = 1LL<<32; // uint_max+1
    return xor128() < prob * x;
}

int timelimit = 10 * 1000, elapsed = 0, margin = 50;

class Timer {
  chrono::system_clock::time_point start_time = chrono::system_clock::now();
public:
  Timer() {}
  int get_elapsed_time() {
    auto diff = chrono::system_clock::now() - start_time;
    return chrono::duration_cast<chrono::milliseconds>(diff).count();
  }
} timer;

// usage:
while (margin + timer.get_elapsed_time() < timelimit) {
    // SA loop
}

// 1D segtree

class SegTree {
    using T = long long;
    static const T& op(const T& lhs, const T& rhs) { // binary operator
        return max(lhs, rhs);
    }
    constexpr static T unit = -1LL<<60; // unit element

    const int N;
    T *data;
    static int calc_size(int sz) {
        int n = 1;
        while (sz > n) n *= 2;
        return n;
    }
public:
    explicit SegTree(int sz) : N(calc_size(sz)) {
        data = new T[2*N-1];
        fill(data, data+2*N-1, unit);
    }
    ~SegTree() { delete[] data; }
    T query(int a, int b, int i = -1, int l = -1, int r = -1) const {
        if (i == -1) { i = 0; l = 0; r = N; }
        if (r <= a || b <= l) return unit; // need this if a query can be empty
        if (a <= l && r <= b) return data[i];
        int m = (l+r)/2;
        if (b <= m) return query(a, b, 2*i+1, l, m);
        if (m <= a) return query(a, b, 2*i+2, m, r);
        T v1 = query(a, b, 2*i+1, l, m),
          v2 = query(a, b, 2*i+2, m, r);
        return op(v1, v2);
    }
    T get(int i) const {
        return data[i + N - 1];
    }
    void update(int i, T v) {
        int x = i + N - 1;
        data[x] = v;
        while (x > 0) {
            x = (x-1)/2;
            data[x] = op(data[2*x+1], data[2*x+2]);
        }
    }
};


// dynamic segtree


class tree {
    static const long long min_idx= 0, max_idx = 1e12;
    struct T {
        double a, b;
    };
    T op(T x, T y) { return {x.a * y.a, x.b * y.a + y.b}; }
    T unit() { return { 1, 0 }; }
 
    tree *left = nullptr, *right = nullptr;
    T prod = unit();
public:
    tree() {}
    void update(long long x, T t, long long l = min_idx, long long r = max_idx+1) {
        if (r - l > 1) {
            if ((l + r) / 2 > x) {
                if (left == nullptr) left = new tree();
                left->update(x, t, l, (l + r) / 2);
            } else {
                if (right == nullptr) right = new tree();
                right->update(x, t, (l + r) / 2, r);
            }
 
            if (left == nullptr) {
                prod = right->prod;
            } else if (right == nullptr) {
                prod = left->prod;
            } else {
                prod = op(left->prod, right->prod);
            }
        } else {
            prod = t;
        }
    }
    T getallprod() { return prod; }
};

class tree {
    constexpr int MAX = 1e9;
    tree *left = nullptr, *right = nullptr;
    int count = 0;
    long long sum = 0;
public:
    void add(int x, int l = -MAX, int r = MAX+1) {
        count++; sum += x;
        if (r - l > 1) {
            if ((l + r) / 2 > x) {
                if (left == nullptr) left = new tree();
                left->add(x, l, (l + r) / 2);
            } else {
                if (right == nullptr) right = new tree();
                right->add(x, (l + r) / 2, r);
            }
        }
    }
    int getnth(int n, int l = -MAX, int r = MAX+1) {
        if (r - l == 1) return l;
        if (left == nullptr) return right->getnth(n, (l+r)/2, r);
        else if (right == nullptr) return left->getnth(n, l, (l+r)/2);
        else if (left->count > n) return left->getnth(n, l, (l+r)/2);
        else return right->getnth(n - left->count, (l+r)/2, r);
    }
    long long getsum(int n, int l = -MAX, int r = MAX+1) {
        if (r - l == 1) return (long long)l * n;
        if (left == nullptr) return right->getsum(n, (l+r)/2, r);
        else if (right == nullptr) return left->getsum(n, l, (l+r)/2);
        else if (left->count > n) return left->getsum(n, l, (l+r)/2);
        else return left->sum + right->getsum(n - left->count, (l+r)/2, r);
    }
};


// 1D segtree with lazy propagation

class SegTreeLazy {
    // monoid and action
    using D = long long int;
    using A = long long int;
    static D unitD() { return numeric_limits<D>::max(); }
    static A unitA() { return 0; }
    static const D compose(const D& lhs, const D& rhs) {
        return min(lhs, rhs);
    }
    static const D applyA(const A& a, const D& x, int l, int r) {
        return a + x;
    }
    static const A appendA(const A& a, const A& b) {
        return a + b;
    }
    // ---

    void force(int i, int l, int r) {
        if (delay[i] != unitA()) {
            data[i] = applyA(delay[i], data[i], l, r);
            if (r - l > 1) {
                delay[2*i+1] = appendA(delay[i], delay[2*i+1]);
                delay[2*i+2] = appendA(delay[i], delay[2*i+2]);
            }
            delay[i] = unitA();
        }
    }
    const int N;
    D *data;
    A *delay;
    int calc_size(int n) { int ret = 1; while (n > ret) ret *= 2; return ret; }
public:
    explicit SegTreeLazy(int n) : N(calc_size(n)) {   // [0, n-1]
        data = new D[2*N-1]();
        delay = new A[2*N-1]();
    }
    ~SegTreeLazy() { delete[] data; delete[] delay; }
    D query(int a, int b, int i = -1, int l = -1, int r = -1) {
        if (i == -1) { i = 0; l = 0; r = N; }
        if (r <= a || b <= l) return unitD();
        force(i, l, r);
        if (a <= l && r <= b) return data[i];
        if (r - l == 1) return data[i];
        const int m = (l+r)/2;
        if (b <= m) return query(a, b, 2*i+1, l, m);
        if (m <= a) return query(a, b, 2*i+2, m, r);
        D v1 = query(a, b, 2*i+1, l, m), v2 = query(a, b, 2*i+2, m, r);
        return compose(v1, v2);
    }
    void add(int a, int b, A v, int i = -1, int l = -1, int r = -1) {
        if (i == -1) { i = 0; l = 0; r = N; }
        if (r <= a || b <= l) return;
        if (a <= l && r <= b) { delay[i] = appendA(v, delay[i]); force(i, l, r); return; }
        add(a, b, v, 2*i+1, l, (l+r)/2); force(2*i+1, l, (l+r)/2);
        add(a, b, v, 2*i+2, (l+r)/2, r); force(2*i+2, (l+r)/2, r);
        data[i] = compose(data[2*i+1], data[2*i+2]);
    }
    void init(D *a, int n) {
        for (int i = 0; i < n; i++) data[i+N-1] = a[i];
        for (int i = N-2; i >= 0; i--) data[i] = compose(data[2*i+1], data[2*i+2]);
    }
    // D get(int i) { return query(i, i+1); }
    // void update(int i, T v) {
    //     int ov = get(i);
    //     add(i, i+1, v-ov); // BEWARE: assumes add is `+'
    // }
};

//     void dump() {
//         cerr << "data "; for (int i = 0; i < 2*N-1; i++) cerr << ' ' << data[i]; cerr << endl;
//         cerr << "delay "; for (int i = 0; i < 2*N-1; i++) cerr << ' ' << delay[i]; cerr << endl;
//     }
// };


// 2D segtree

class segtree2d {
    int H, W;
    int Nx, Ny;
public:
    vector<vector<int> > data;
    segtree2d(int H0, int W0) {
        H = H0; W = W0;
        Nx = 1; while (Nx < H) Nx *= 2;
        Ny = 1; while (Ny < W) Ny *= 2;
        data = vector<vector<int> >(2*Nx-1, vector<int>(2*Ny-1, 0));
    }
    int query(int ax, int ay, int bx, int by,
              int i = 0, int j = 0, int lx = 0, int ly = 0, int rx = -1, int ry = -1) {
        if (rx == -1) { rx = Nx; ry = Ny; }
        if (rx <= ax || bx <= lx) return 0;
        if (ax <= lx && rx <= bx) return query_y(ax, ay, bx, by, i, j, lx, ly, rx, ry);
        int v1 = query(ax, ay, bx, by, 2*i+1, j, lx, ly, (lx+rx)/2, ry),
            v2 = query(ax, ay, bx, by, 2*i+2, j, (lx+rx)/2, ly, rx, ry);
        return v1 + v2;
    }
    int query_y(int ax, int ay, int bx, int by, int i, int j, int lx, int ly, int rx, int ry) {
        if (ry <= ay || by <= ly) return 0;
        if (ay <= ly && ry <= by) return data[i][j];
        int v1 = query_y(ax, ay, bx, by, i, 2*j+1, lx, ly, rx, (ly+ry)/2),
            v2 = query_y(ax, ay, bx, by, i, 2*j+2, lx, (ly+ry)/2, rx, ry);
        return v1 + v2;
    }
    void update(int x0, int y0, int v) {
        int x = x0 + Nx - 1;
        int y = y0 + Ny - 1;
        data[x][y] = v;
        while (y > 0) {
            y = (y-1)/2;
            data[x][y] = data[x][2*y+1] + data[x][2*y+2];
        }
        while (x > 0) {
            x = (x-1)/2;
            y = y0 + Ny - 1;
            data[x][y] = data[2*x+1][y] + data[2*x+2][y];
            while (y > 0) {
                y = (y-1)/2;
                data[x][y] = data[x][2*y+1] + data[x][2*y+2];
            }
        }
    }
};

// MST
vector<vector<pair<int, int> > > temp(n);
bool visited[n] = {};
visited[0] = true;
priority_queue<tuple<int, int, int> > pq; // cost, src, dst
int depth[n] = {};
for (auto e : adj[0]) pq.push(make_tuple(-e.second, 0, e.first));
while (!pq.empty()) {
    auto e = pq.top(); pq.pop();
    int cost = -get<0>(e), src = get<1>(e), dst = get<2>(e);
    if (visited[dst]) continue;
    visited[dst] = true;
    depth[dst] = depth[src] + 1;
    temp[src].push_back({dst, cost});
    sum += cost;
    for (auto e : adj[dst])
        if (!visited[e.first]) pq.push(make_tuple(-e.second, dst, e.first));
}
temp.swap(adj);


// RMQ
class SparseTableRMQ {
    using T = int;
    static const T& op(const T& lhs, const T& rhs) { // binary operator
        return max(lhs, rhs);
    }
    constexpr static T unit = 0; // unit element
    int N, K;
    T **tbl;
    void delete_tbl() {
        if (tbl != nullptr) {
            for (int k = 0; k < K; k++)
                if (tbl[k] != nullptr) delete[] tbl[k];
            delete[] tbl;
        }
    }
public:
    void init(int n, T *p) {
        delete_tbl();
        N = n;
        K = 0;
        while (1<<K <= n) K++;
        tbl = new T*[K];
        for (int k = 0; k < K; k++) {
            tbl[k] = new T[N];
            int w = 1<<k;
            for (int i = 0; i+w <= N; i++)
                if (k == 0) tbl[0][i] = p[i];
                else tbl[k][i] = op(tbl[k-1][i], tbl[k-1][i+w/2]);
        }
    }
    ~SparseTableRMQ() { delete_tbl(); }
    T query (int l, int r) const {
        int d = r - l;
        if (d <= 0) return unit;
        int lb = 0, ub = K;
        while (ub - lb > 1) {
            int m = (ub + lb) / 2;
            if (1<<m <= d) lb = m;
            else ub = m;
        }
        return op(tbl[lb][l], tbl[lb][r-(1<<lb)]);
    }
};

// slide minmax
template<typename T, class Compare = less<T> >
class SlideMin {
    const int K;
    int i = 0;
    deque<pair<T, int> > data;
public:
    SlideMin(int k) : K(k) {};
    T get() { return data.front().first; }
    void put(T x) {
        while (!data.empty() && Compare()(x, data.back().first)) data.pop_back();
        if (!data.empty() && data.front().second == i-K) data.pop_front();
        data.push_back({x, i++});
    }
    // void debug() { for (auto x : data) cerr << x.first << ' '; cerr << endl; }
};


// Ford-Fulkerson (w/o construction)
class FordFulkerson {
    int S, T;
    struct Edge {
        int dst, cap, rev;
    };
    vector<vector<Edge> > adj;
    vector<bool> visited;
    const int INF = 1<<30;
    int dfs(int i, int f) {
        visited[i] = true;
        if (i == T) return f;
        for (Edge &e : adj[i]) {
            if (!visited[e.dst] && e.cap > 0) {
                int r = dfs(e.dst, min(e.cap, f));
                if (r > 0) {
                    e.cap -= r;
                    adj[e.dst][e.rev].cap += r;
                    return r;
                }
            }
        }
        return 0;
    }
public:
    FordFulkerson(int n, int s, int t) : S(s), T(t) {
        adj.resize(n);
        visited.resize(n);
    }
    void add_edge(int u, int v) {
        int i = adj[v].size(), j = adj[u].size();
        adj[u].push_back({v, 1, i});
        adj[v].push_back({u, 0, j});
    }
    int get() {
        int f = 0;
        for (;;) {
            fill(visited.begin(), visited.end(), false);
            int r = dfs(S, INF);
            if (r > 0) f += r; else return f;
        }
    }
};


// closest pair
using P = pair<int, int>;

long long dist2(P a, P b) {
    long long dx = a.first - b.first, dy = a.second - b.second;
    return dx * dx + dy * dy;
}

// returns square of min. dist.
long long closest(P* p, P* q) { 
    if (q - p <= 1) return 9e18;
    P* r = p + (int)(q - p) / 2;
    int m = r->first;
    long long d2 = min(closest(p, r), closest(r, q)), ans2 = d2;
    inplace_merge(p, r, q, [&](P a, P b) { return a.second < b.second; });
    vector<P> ps;
    for (; p < q; p++) {
        long long dx = p->first - m;
        if (dx * dx > d2) continue;
        for (int i = ps.size()-1; i >= 0; i--) {
            long long dy = p->second - ps[i].second;
            if (dy * dy > d2) break;
            ans2 = min(ans2, dist2(*p, ps[i]));
        }
        ps.push_back(*p);
    }
    return ans2;
}


// convex hull, farthest pair, peremeter
using P = pair<int, int>;
P operator-(P& lhs, P& rhs) {
    return {lhs.first - rhs.first, lhs.second - rhs.second};
}

double dist(P a, P b) {
    double dx = a.first - b.first, dy = a.second - b.second;
    return sqrt(dx * dx + dy * dy);
}

long long cp(P a, P b) {
    return (long long)a.first * b.second - (long long)a.second * b.first;
}

vector<int> convex_hull(P* p, int n) {
    vector<int> ps(n);
    iota(ps.begin(), ps.end(), 0);
    sort(ps.begin(), ps.end(), [&](int i, int j) { return p[i] < p[j]; });
    vector<int> lower;
    for (int i : ps) {
        while (lower.size() > 1) {
            const int k = lower.size();
            P cur = p[i], prev = p[lower[k-1]], pprev = p[lower[k-2]];
            if (cp(prev - pprev, cur - pprev) > 0) break;
            else lower.pop_back();
        }
        lower.push_back(i);
    }
    reverse(ps.begin(), ps.end());
    vector<int> upper;
    for (int i : ps) {
        while (upper.size() > 1) {
            const int k = upper.size();
            P cur = p[i], prev = p[upper[k-1]], pprev = p[upper[k-2]];
            if (cp(prev - pprev, cur - pprev) > 0) break;
            else upper.pop_back();
        }
        upper.push_back(i);
    }
    lower.insert(lower.end(), upper.begin()+1, upper.end());
    return lower;
}

double farthest_dist(P* p, int n) {
    vector<int> v = convex_hull(p, n);
    double ret = 0;
    const int m = v.size();
    for (int i = 0, j = 0; i < m; i++) {
        while (dist(p[v[i]], p[v[j]]) <= dist(p[v[i]], p[v[(j+1)%m]]))
            j = (j + 1) % m;
        ret = max(ret, dist(p[v[i]], p[v[j]]));
    }
    return ret;
}

double perimeter(P* p, int n) {
    vector<int> v = convex_hull(p, n);
    double ret = 0;
    for (int i = 1; i < v.size(); i++)
        ret += dist(p[v[i-1]], p[v[i]]);
    return ret;
}


// printing __int128
ostream &operator<<(ostream &out, __int128 x) {
    bool minus = x < 0;
    if (x < 0) x = -x;
    constexpr int bufsize = 42;
    char buf[bufsize], *p = &buf[bufsize-1];
    *p = '\0';
    while (x > 0) {
        *--p = (x % 10) + '0';
        x /= 10;
    }
    if (minus) out << '-';
    out << p;
    return out;
}

// bitDP 3^n subset enumeration
// t \subset s \subset {0..n-1}
for (int s = 0; s < 1<<n; s++) {
    for (int t = s; ; t = (t - 1) & s) {
        // ...
        if (t == 0) break;
    }
}


// FFT
using C = complex<double>;
const double PI = acos(-1);

void fft(vector<C> &a, bool inv) {
    int n = a.size();
    if (n == 1) return;

    vector<C> a0(n / 2), a1(n / 2);
    for (int i = 0; 2 * i < n; i++) {
        a0[i] = a[2*i];
        a1[i] = a[2*i+1];
    }
    fft(a0, inv);
    fft(a1, inv);

    const double ang = 2 * PI / n * (inv ? -1 : 1);
    const C wn(cos(ang), sin(ang));
    C w(1);
    for (int i = 0; 2 * i < n; i++) {
        a[i] = a0[i] + w * a1[i];
        a[i + n/2] = a0[i] - w * a1[i];
        if (inv) {
            a[i] /= 2;
            a[i + n/2] /= 2;
        }
        w *= wn;
    }
}

vector<long long> multiply(vector<int> const& a, vector<int> const& b) {
    vector<C> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    int n = 1;
    while (n < a.size() + b.size()) 
        n <<= 1;
    fa.resize(n);
    fb.resize(n);

    fft(fa, false);
    fft(fb, false);
    for (int i = 0; i < n; i++)
        fa[i] *= fb[i];
    fft(fa, true);

    vector<long long> result(n);
    for (int i = 0; i < n; i++)
        result[i] = round(fa[i].real());
    return result;
}


// // trie
// class Node {
// public:
//     int chidx[26];
//     bool exist = false;
//     Node() {
//     }
//     int child(char c) {
//         return chidx[c - 'a'];
//     }
// };

// class Trie {
//     vector<Node> pool(1);
// public:
//     void add(string &s) {
//         int idx = 0;
//         for (char c : s) {
//             int nxt = pool[idx].child(c);
//             if (nxt == -1) {
//                 nxt = pool.size();
//                 pool[idx].child(c) = nxt;
//                 pool.push_back(Node());
//             }
//             idx = nxt;
//         }
//         pool[idx].exist = true;
//     }
// }


// patricia
// https://atcoder.jp/contests/code-festival-2016-qualb/submissions/15983585
struct Tree {
    struct Node {
        static int serial_id;
        int id;
        bool exists = false;
        int count = 0;
        Node *children[26];
        string prefix[26];
        Node() {
            fill(children, children+26, nullptr);
            id = serial_id++;
        }
    };

    Node *root;
    Tree() { root = new Node(); }
    void add(string &s, int i=0, Node *np=nullptr) {
        if (np == nullptr) np = root;
        np->count++;
        if (np->children[s[i]-'a'] == nullptr) {
            Node *ch = new Node();
            np->children[s[i]-'a'] = ch;
            np->prefix[s[i]-'a'] = s.substr(i);
            ch->exists = true;
            ch->count = 1;
        } else {
            Node *ch = np->children[s[i] - 'a'];
            string &pref = np->prefix[s[i] - 'a'];
            int cp = 0;
            while (i + cp < s.size() && cp < pref.size()) {
                if (s[i + cp] != pref[cp]) break;
                else cp++;
            }
            if (cp == pref.size()) {
                if (i + cp == s.size()) {
                    ch->exists = true;
                    ch->count++;
                } else
                    add(s, i + cp, ch);
            } else {
                Node *mid = new Node();
                mid->children[pref[cp] - 'a'] = ch;
                mid->prefix[pref[cp] - 'a'] = pref.substr(cp);
                np->children[s[i] - 'a'] = mid;
                np->prefix[s[i] - 'a'] = s.substr(i, cp);
                mid->count = 1 + ch->count;
                if (i + cp == s.size()) {
                    mid->exists = true;
                } else {
                    Node *leaf = new Node();
                    leaf->exists = true;
                    leaf->count = 1;
                    mid->children[s[i + cp] - 'a'] = leaf;
                    mid->prefix[s[i + cp] - 'a'] = s.substr(i + cp);
                }
            }
        }
    }

    int calc_rank(const string &s, const string &p) {
        int ans = 0;
        Node *nd = root;
        for (int i = 0; i < s.size();) {
            // cerr << s << " " << i << " " << nd->id << endl;
            for (int j = 0; s[i] != p[j]; j++) {
                Node *ch = nd->children[p[j] - 'a'];
                if (ch != nullptr) ans += ch->count;
            }
            Node *ch = nd->children[s[i] - 'a'];
            i += nd->prefix[s[i] - 'a'].size();
            nd = ch;
            if (nd->exists) ans++;
        }
        return ans;
    }

    void dump(Node *nd=nullptr) {
        if (nd == nullptr) nd = root;
        cerr << "Node id = " << nd->id << endl;
        cerr << (nd->exists ? "Exist" : "Not exist") << "; count = " << nd->count << endl;
        cerr << "children = ";
        for (int i = 0; i < 26; i++) {
            Node *ch = nd->children[i];
            cerr << " " << (ch == nullptr ? -1 : ch->id);
        }
        cerr << endl;
        cerr << "prefix = ";
        for (int i = 0; i < 26; i++) {
            cerr << "; " << nd->prefix[i];
        }
        cerr << endl;
        for (int i = 0; i < 26; i++) {
            Node *ch = nd->children[i];
            if (ch != nullptr) dump(ch);
        }
    }
};


// bipartite matching; slower than atcoder lib.
class BipartiteMatching {
    const int N, M;
    vector<vector<int> > adj;
    vector<int> match;
    vector<bool> visited;
    bool dfs(int i) {
        visited[i] = true;
        for (int j : adj[i]) {
            if (visited[j]) continue;
            visited[j] = true;
            if (match[j] == -1 || dfs(match[j])) {
                match[i] = j;
                match[j] = i;
                return true;
            }
        }
        return false;
    }
public:
    BipartiteMatching(int n, int m) : N(n), M(m) {
        adj.resize(m+n);
        match.resize(m+n, -1);
        visited.resize(m+n);
    }
    void add_edge(int i, int j) {
        assert(0 <= i);
        assert(i < N);
        assert(0 <= j);
        assert(j < M);
        adj[i].push_back(j+N);
        adj[j+N].push_back(i);
    }
    vector<pair<int, int> > matching() {
        vector<pair<int, int> > result;
        for (;;) {
            bool found = false;
            fill(visited.begin(), visited.end(), false);
            for (int i = 0; i < N; i++) {
                if (match[i] >= 0) continue;
                if (dfs(i)) { found = true; break; }
            }
            if (!found) break;
        }
        for (int i = 0; i < N; i++)
            if (match[i] >= 0) result.push_back({i, match[i]-N});
        return result;
    }
};


// Z algorithm
vector<int> z(string &s) {
    const int n = s.size();
    vector<int> ret(n, 0);
    ret[0] = n;
    int l, r = 1;
    for (int i = 1; i < n; i++) {
        if (i >= r) {
            int j = 0;
            while (i + j < n && s[j] == s[i + j]) j++;
            l = i; r = i + j;
            ret[i] = j;
        } else {
            int k = ret[i - l];
            if (k < r - i) ret[i] = k;
            else {
                int j = r - i;
                while (i + j < n && s[j] == s[i + j]) j++;
                l = i; r = i + j;
                ret[i] = j;
            }
        }
    }
    return ret;
}


// 関節点 articulation points
vector<int> articulation_points(vector<vector<int>> &adj) {
    const int n = adj.size();
    vector<int> disc(n, -1), low(n);
    vector<int> aps;
    int counter = 0;
    function<void(int, int)> dfs = [&](int i, int p) {
        disc[i] = low[i] = counter++;
        bool is_ap = false;
        int ccount = 0;
        for (int j : adj[i]) {
            if (j == p) continue;
            if (disc[j] == -1) {
                dfs(j, i);
                low[i] = min(low[i], low[j]);
                ccount++;
                if (p >= 0 && low[j] >= disc[i]) is_ap = true;
            } else {
                low[i] = min(low[i], disc[j]);
            }
        }
        if (is_ap || p == -1 && ccount > 1) aps.push_back(i);
    };
    for (int i = 0; i < n; i++) if (disc[i] == -1) dfs(i, -1);
    sort(aps.begin(), aps.end());
    return aps;
}


// bridges
vector<pair<int, int> > bridges(vector<vector<int>> &adj) {
    const int n = adj.size();
    vector<int> disc(n, -1), low(n);
    vector<pair<int, int> > bs;
    int counter = 0;
    function<void(int, int)> dfs = [&](int i, int p) {
        disc[i] = low[i] = counter++;
        bool is_ap = false;
        int ccount = 0;
        for (int j : adj[i]) {
            if (j == p) continue;
            if (disc[j] == -1) {
                dfs(j, i);
                low[i] = min(low[i], low[j]);
                ccount++;
            } else {
                low[i] = min(low[i], disc[j]);
            }
            if (low[j] > disc[i]) bs.emplace_back(i, j);
        }
    };
    for (int i = 0; i < n; i++) if (disc[i] == -1) dfs(i, -1);
    return bs;
}

// SCC
// returns: node -> index of its component
vector<int> scc(const vector<vector<int>> &adj) {
    const int n = adj.size();
    vector<int> low(n), disc(n, -1), comp(n);
    vector<bool> finished(n, false);
    stack<int> st;
    int count = 0, cidx = 0;
    function<void(int)> dfs = [&](int i) {
        low[i] = disc[i] = count++;
        st.push(i);
        for (int j : adj[i]) {
            if (disc[j] == -1) {
                dfs(j);
                low[i] = min(low[i], low[j]);
            } else if (!finished[j]) {
                low[i] = min(low[i], disc[j]);
            }
        }

        if (low[i] == disc[i]) {
            for (;;) {
                int x = st.top();
                st.pop();
                finished[x] = true;
                comp[x] = cidx;
                if (x == i) break;
            }
            cidx++;
        }
    };
    for (int i = 0; i < n; i++) if (disc[i] == -1) dfs(i);

    return comp;
}


// fast zeta/Moebius transform
template<typename T, bool dual=false, bool inverse=false>
void zeta_transform(vector<T>& f) {
    const int n = f.size();
    for (int i = 0; (1 << i) < n; i++)
        for (int s = 0; s < n; s++)
            if constexpr (dual) {
                if (!((s >> i) & 1)) {
                    if constexpr (inverse) f[s] = f[s] - f[s^(1<<i)];
                    else f[s] = f[s] + f[s^(1<<i)];
                }
            } else {
                if ((s >> i) & 1) {
                    if constexpr (inverse) f[s] = f[s] - f[s^(1<<i)];
                    else f[s] = f[s] + f[s^(1<<i)];
                }
            }
}

template<typename T>
void (*subset_zeta)(vector<T>&) = zeta_transform<T, false, false>;

template<typename T>
void (*supset_zeta)(vector<T>&) = zeta_transform<T, true, false>;

template<typename T>
void (*subset_moebius)(vector<T>&) = zeta_transform<T, false, true>;

template<typename T>
void (*supset_moebius)(vector<T>&) = zeta_transform<T, true, true>;

// check
/*
set<int> op(set<int>& a, set<int>& b) {
    set<int> ret = b;
    for (int x : a) ret.insert(x);
    return ret;
}

set<int> inv(set<int>& a, set<int>& b) {
    set<int> ret = a;
    for (int x : b) ret.erase(x);
    return ret;
}

class TestVal : public set<int> {

};

TestVal operator+(TestVal& lhs, TestVal& rhs) {
    TestVal ret = lhs;
    for (int x : rhs) ret.insert(x);
    return ret;
}

TestVal operator-(TestVal& lhs, TestVal& rhs) {
    TestVal ret = lhs;
    for (int x : rhs) ret.erase(x);
    return ret;
}

void test1(int n) {
    vector<TestVal> v(1<<n);
    for (int i = 0; i < v.size(); i++) v[i].insert(i);
    subset_zeta<TestVal>(v);
    for (int i = 0; i < v.size(); i++)
        for (int j = 0; j < v.size(); j++)
            assert((v[i].find(j) != v[i].end()) == ((i & j) == j));

    subset_moebius<TestVal>(v);
    for (int i = 0; i < v.size(); i++)
        for (int j = 0; j < v.size(); j++)
            assert((v[i].find(j) != v[i].end()) == (i == j));
}

void test2(int n) {
    vector<TestVal> v(1<<n);
    for (int i = 0; i < v.size(); i++) v[i].insert(i);
    supset_zeta<TestVal>(v);
    for (int i = 0; i < v.size(); i++)
        for (int j = 0; j < v.size(); j++)
            assert((v[i].find(j) != v[i].end()) == ((i & j) == i));

    supset_moebius<TestVal>(v);
    for (int i = 0; i < v.size(); i++)
        for (int j = 0; j < v.size(); j++)
            assert((v[i].find(j) != v[i].end()) == (i == j));
}

int main() {
    for (int i = 1; i < 12; i++) {
        test1(i);
        test2(i);
    }
}
*/

// determinant of matrix m mod M; m is modified
int det_mod(vector<vector<int> >& m) {
    long long result = 1;
    const int n = m.size();
    for (int i = 0; i < n; i++) {
        int j = i;
        while (j < n && m[j][i] == 0) j++;
        if (j == n) return 0;
        m[i].swap(m[j]);
        result = result * m[i][i] % M;
        const long long mult = modinv(m[i][i], M);
        for (int k = i + 1; k < n; k++) {
            if (const int v = m[k][i]; v != 0) {
                for (int j = i; j < n; j++) {
                    m[k][j] -= m[i][j] * mult % M * v % M;
                    m[k][j] %= M;
                    if (m[k][j] < 0) m[k][j] += M;
                }
            }
        }
    }
    return result;
}
