#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Installing..."

cd $DIR/spn
rm -r ./build
luarocks make spn-scm-1.rockspec

cd $DIR/cuspn
rm -r ./build
luarocks make cuspn-scm-1.rockspec

echo "Done!"