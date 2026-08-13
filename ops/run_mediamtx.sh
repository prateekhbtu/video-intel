#!/usr/bin/env bash
while true; do
  /usr/local/bin/mediamtx /etc/mediamtx/mediamtx.yml >> "${DATA}/logs/mediamtx.log" 2>&1
  echo "mediamtx exited, restarting" >> "${DATA}/logs/mediamtx.log"
  sleep 2
done
