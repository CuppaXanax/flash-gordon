set -eu
d=/home/xander/hc-abq-stack-ad343506d21c49e8a3a112f98d2eb602
date -Is
if test -f "$d/coordinator.log"; then tail -n 5 "$d/coordinator.log"; fi
if test -d "$d/logits"; then find "$d/logits" -maxdepth 1 -type f | wc -l; fi
