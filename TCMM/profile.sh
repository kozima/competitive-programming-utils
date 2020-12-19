NAME=

if which pprof; then
    if test ${NAME}.cpp -nt ${NAME}_prof; then
        g++ -g --std=gnu++11 -O3 -o ${NAME}_prof -lprofiler -Wl,-no_pie ${NAME}.cpp
    fi
    env CPUPROFILE=${NAME}.prof ./${NAME}_prof 10 -b < tests/2.in > /dev/null
    pprof --text ${NAME}_prof ${NAME}.prof
elif which gprof; then
    if test ! -e ${NAME}_prof || test ${NAME}.cpp -nt ${NAME}_prof; then
        g++ -pg --std=c++14 -lm -x c++ -O2 -o ${NAME}_prof ${NAME}.cpp
    fi

    ./${NAME}_prof 10 < tests/2.in > /dev/null

    gprof ${NAME}_prof
fi
