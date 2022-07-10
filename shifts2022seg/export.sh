#!/usr/bin/env bash

./build.sh

docker save shifts2022seg | gzip -c > shifts2022seg.tar.gz
