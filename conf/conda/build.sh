#!/bin/bash

if [ "$(uname)" == "Darwin" ]
then

    export CXX=mpicxx
    export CXXFLAGS="-stdlib=libc++ ${CXXFLAGS}"
    export LDFLAGS="-Wl,-rpath,$PREFIX/lib"
fi

$PYTHON setup.py install --prefix=$PREFIX
