#!/bin/sh
set -eu
case " $(hostname -I) " in *" 192.168.42.50 "*) ;; *) exit 64;; esac
systemctl is-active bc250-embed bc250-rerank
systemctl show -p MainPID bc250-embed bc250-rerank
curl -fsS --max-time 10 http://192.168.42.50:8091/health
printf '\n'
curl -fsS --max-time 10 http://192.168.42.50:8092/health
printf '\n'
