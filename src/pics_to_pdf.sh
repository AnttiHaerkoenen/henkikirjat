#!/usr/bin/env bash

cd ../data
pwd
sudo convert *.jpg +compress esim.pdf
echo pdf created
sudo chown -R antth esim.pdf
echo ownership changed