name=

if which pprof; then
    if test ${name}.cpp -nt ${name}_prof; then
        g++ -g --std=c++14 -lm -x c++ -O2 -o ${name}_prof -lprofiler -Wl,-no_pie ${name}.cpp
    fi
    env CPUPROFILE=${name}.prof ./${name}_prof 10 -b < testcases/1.txt > /dev/null
    pprof --text ${name}_prof ${name}.prof
elif which gprof; then
    if test ! -e ${name}_prof || test ${name}.cpp -nt ${name}_prof; then
        g++ -pg --std=c++14 -lm -x c++ -O2 -o ${name}_prof ${name}.cpp
    fi

    ./${name}_prof 10 < input/2.txt > /dev/null #dense
    # ./${name}_prof 10 < testcases/10.txt > /dev/null #sparse

    gprof ${name}_prof
fi
