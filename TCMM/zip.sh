NAME=

mkdir -p submissions
zipfile=`date +%m%d_%H%M%S`.zip
zip submissions/$zipfile $NAME.cpp
