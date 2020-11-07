#include <cassert>
#include <chrono>

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

inline bool rand_bool() {
    return xor128() < 1u<<31;
}

inline double rand(double lb, double ub) {
    assert(lb < ub);
    unsigned int x = xor128();
    return lb + (ub - lb) * x / double(1LL<<32);
}

inline int rand_int(int n) { return xor128()%n; }

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

