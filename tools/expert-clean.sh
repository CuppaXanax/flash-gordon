#!/bin/sh
set -eu
case " $(hostname -I) " in *" 192.168.42.50 "*) ;; *) exit 64;; esac
test -d /home/xander/hc-experts
rm -rf /home/xander/hc-experts
test ! -e /home/xander/hc-experts
printf 'hc-experts removed\n'
systemctl is-active bc250-embed bc250-rerank
curl -fsS --max-time 10 http://192.168.42.50:8091/health
printf '\n'
curl -fsS --max-time 10 http://192.168.42.50:8092/health
printf '\n'
