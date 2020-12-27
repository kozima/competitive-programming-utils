#include <iostream>
#include <tuple>
#include <unordered_map>
#include <vector>
using namespace std;

namespace BDD {

constexpr int TRUE = -1, FALSE = -2;
constexpr int to_node_id(bool b) { return int(b ? TRUE : FALSE); }

static_assert(to_node_id(true) == TRUE);
static_assert(to_node_id(false) == FALSE);

struct Node {
    const int var;
    Node * const c0 = nullptr, * const c1 = nullptr;
    Node(bool b) : var(to_node_id(b)) {}
    Node(int v, Node *c0, Node *c1) : var(v), c0(c0), c1(c1) {}
    bool is_terminal_node() const { return var < 0; }
    bool is_decision_node() const { return var >= 0; }
    bool is_true_node() const { return var == TRUE; }
    bool is_false_node() const { return var == FALSE; }
};

Node* true_node = new Node(true);
Node* false_node = new Node(false);

class Node_pool {
    vector<Node*> v;
public:
    Node_pool() {}
    ~Node_pool() { for (Node* p : v) delete p; }
    void add(Node *p) { v.push_back(p); }
} pool;

Node *create_node(int v, Node *d0, Node *d1) {
    Node *p = new Node(v, d0, d1);
    pool.add(p);
    return p;
}

struct lookup_hash {
    size_t operator()(const tuple<int, Node *, Node *>& t) const {
        return hash<int>()(get<0>(t)) + hash<Node *>()(get<1>(t)) + hash<Node *>()(get<2>(t));
    }
};

unordered_map<tuple<int, Node *, Node *>, Node *, lookup_hash> lookup_table;

Node *merge(int v, Node *d0, Node *d1) {
    if(auto it = lookup_table.find(make_tuple(v, d0, d1)); it != lookup_table.end())
        return it->second;
    else {
        Node *p = create_node(v, d0, d1);
        lookup_table.emplace(make_tuple(v, d0, d1), p);
        return p;
    }
}

Node *And(Node * a, Node * b) {
    if (a->is_false_node() || b->is_true_node()) return a;
    if (a->is_true_node() || b->is_false_node()) return b;
    if (a->var == b->var) {
        return merge(a->var, And(a->c0, b->c0), And(a->c1, b->c1));
    }
    if (a->var > b->var) swap(a, b);
    return merge(a->var, And(a->c0, b), And(a->c1, b));
}

Node *Or(Node * a, Node * b) {
    if (a->is_true_node() || b->is_false_node()) return a;
    if (a->is_false_node() || b->is_true_node()) return b;
    if (a->var == b->var) {
        return merge(a->var, Or(a->c0, b->c0), Or(a->c1, b->c1));
    }
    if (a->var > b->var) swap(a, b);
    return merge(a->var, Or(a->c0, b), Or(a->c1, b));
}

Node *Not(Node * a) {
    if (a->is_true_node()) return false_node;
    if (a->is_false_node()) return true_node;
    return merge(a->var, Not(a->c0), Not(a->c1));
}

Node *Var(int v) { return merge(v, false_node, true_node); }
Node *NegVar(int v) { return merge(v, true_node, false_node); }

}

long long count(BDD::Node * a, int v_min, int v_max) {
    if (a->is_true_node()) return 1LL << (v_max - v_min);
    if (a->is_false_node()) return 0;
    int k = a->var - v_min;
    return (count(a->c0, a->var + 1, v_max) + count(a->c1, a->var + 1, v_max)) << k;
};

vector<string> elements(BDD::Node * a, int v_min, int v_max) {
    if (a->is_true_node()) {
        vector<string> result;
        for (int s = 0; s < 1<<(v_max - v_min); s++) {
            string &t = result.emplace_back();
            for (int i = 0; i < (v_max - v_min); i++)
                t.push_back(((s >> i) & 1) ? '1' : '0');
        }
        return result;
    }
    if (a->is_false_node()) return {};
    if (a->var == v_min) {
        vector<string> res0 = elements(a->c0, v_min + 1, v_max),
            res1 = elements(a->c1, v_min + 1, v_max),
            result;
        for (string &s : res0) result.emplace_back("0" + s);
        for (string &s : res1) result.emplace_back("1" + s);
        return result;
    } else {
        vector<string> result;
        for (string &s : elements(a, v_min + 1, v_max)) {
             result.emplace_back("0" + s);
             result.emplace_back("1" + s);
        }
        return result;
    }
};

int main() {
    using namespace BDD;
    Node *p = true_node;
    // for (string s : elements(p, 0, 0)) cout << s << ' '; cout << endl;
    // cout << elements(Var(0), 0, 1).size() << endl;
    // cout << elements(NegVar(0), 0, 1).size() << endl;
    // for (string s : elements(NegVar(0), 0, 1)) cout << s << ' '; cout << endl;
    // for (string s : elements(p, 0, 3)) cout << s << ' '; cout << endl;
    p = And(Var(0), Or(Var(1), Var(2)));
    for (string s : elements(p, 0, 3)) cout << s << ' '; cout << endl;
    for (string s : elements(p, 0, 4)) cout << s << ' '; cout << endl;
    for (string s : elements(Not(p), 0, 3)) cout << s << ' '; cout << endl;
    for (string s : elements(Not(p), 0, 4)) cout << s << ' '; cout << endl;
    cout << count(p, 0, 3) << endl;
    cout << count(p, 0, 4) << endl;
    cout << count(Not(p), 0, 3) << endl;
    cout << count(Not(p), 0, 4) << endl;
    cout << count(And(p, p), 0, 4) << endl;
}
