#!/bin/bash

if [ "$(uname)" == "Darwin" ]
then
    export CXXFLAGS="-stdlib=libc++ ${CXXFLAGS}"
    export LDFLAGS="-Wl,-rpath,$PREFIX/lib"
fi

pip install --no-deps --no-binary :all: -r "${RECIPE_DIR}/component-requirements.txt"

$PYTHON -m pip install . --no-deps --ignore-installed --no-cache-dir -vvv
