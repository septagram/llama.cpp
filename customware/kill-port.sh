#!/usr/bin/env bash
# Stop the llama-server that listens on one port, and nothing else: the daily launcher on 5000 must survive a test
# on 5011. `taskkill //IM llama-server.exe` killed every server on the box, including the user's.
#   usage: customware/kill-port.sh <port>
port=${1:?port}
pids=$(netstat -ano -p tcp 2>/dev/null | awk -v p=":$port" '$2 ~ p"$" && $4 == "LISTENING" { print $5 }' | sort -u)
if [ -z "$pids" ]; then
  echo "kill-port: nothing listens on $port"
  exit 0
fi
for pid in $pids; do
  name=$(tasklist //FI "PID eq $pid" //FO CSV //NH 2>/dev/null | head -1 | cut -d, -f1 | tr -d '"')
  if [ "$name" = "llama-server.exe" ]; then
    taskkill //F //PID "$pid" > /dev/null 2>&1 && echo "kill-port: stopped llama-server pid $pid on $port"
  else
    echo "kill-port: pid $pid on $port is '$name', not llama-server: left alone"
  fi
done
