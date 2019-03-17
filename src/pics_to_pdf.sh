#!/usr/bin/env bash

cd ../data
pwd
sudo convert *.jpg +compress esim.pdf
echo Pdf created
sudo chown -R antth esim.pdf
echo Ownership changed