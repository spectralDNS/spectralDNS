#!/bin/bash

if [ "$(uname)" == "Darwin" ]
then
    export CXXFLAGS="-stdlib=libc++ ${CXXFLAGS}"
    export LDFLAGS="-Wl,-rpath,$PREFIX/lib"
fi

pip install --no-deps --no-binary :all: -r "${RECIPE_DIR}/component-requirements.txt"
pip install pythran || echo "no pythran support"

$PYTHON setup.py build_ext --inplace
$PYTHON setup.py install --prefix=$PREFIX
