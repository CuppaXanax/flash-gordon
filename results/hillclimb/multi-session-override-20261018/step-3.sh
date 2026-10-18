set -euo pipefail
command -v setpriv
remaining=$((1792363859-$(date +%s)))
test "$remaining" -gt 300
systemd-run --unit=flash-gordon-override-deadline --on-active="${remaining}s" --property=UMask=0077 --property=KillMode=process --property=LimitMEMLOCK=infinity /usr/bin/bash -c "$(printf '%s' 'L3Vzci9iaW4vc2V0cHJpdiAtLXJldWlkPXhhbmRlciAtLXJlZ2lkPXhhbmRlciAtLWluaXQtZ3JvdXBzIC91c3IvYmluL2VudiBIT01FPS9ob21lL3hhbmRlciBVU0VSPXhhbmRlciAvdXNyL2Jpbi9weXRob24zICcvaG9tZS94YW5kZXIvZmxhc2gtZ29yZG9uLWxpdmUvMjAyNjEwMTUtYWJxLW9wZXJhdG9yL2ZnX2xlYXNlX2RlYWRsaW5lLnB5JyAtLWNvbmZpZyAnL2hvbWUveGFuZGVyL2ZsYXNoLWdvcmRvbi1saXZlLzIwMjYxMDE1LWFicS1vcGVyYXRvci9zdXBlcnZpc29yLmpzb24nOyBzdGF0dXM9JD87IC91c3IvYmluL3N5c3RlbWN0bCBzdGFydCBmbGFzaC1nb3Jkb24tcmluZy5zZXJ2aWNlOyBleGl0ICRzdGF0dXM=' | base64 -d)"
systemctl is-active flash-gordon-override-deadline.timer