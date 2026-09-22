#!/bin/bash
# scan_release_leaks.sh — full leak scan over the STAGED release bytes (A398 precondition).
# Usage: bash ops/scan_release_leaks.sh <staged_dir>
# Exit 1 on any BLOCK hit (release refused). REVIEW hits are printed for a human eyeball pass
# (they are allowed only inside explicit disclaimers/negations — verify, don't assume).
set -u
DIR="${1:?usage: scan_release_leaks.sh <staged_dir>}"
[ -d "$DIR" ] || { echo "no such dir: $DIR"; exit 2; }

block() { # name, extended-regex
  local name="$1" pat="$2"
  grep -rInE --binary-files=without-match "$pat" "$DIR" 2>/dev/null | grep -v "scan_release_leaks" > /tmp/_leak_hits.$$ || true
  if [ -s /tmp/_leak_hits.$$ ]; then
    echo "🚫 BLOCK — $name:"
    head -20 /tmp/_leak_hits.$$
    BLOCKED=1
  fi
}
review() {
  local name="$1" pat="$2"
  grep -rInE --binary-files=without-match "$pat" "$DIR" 2>/dev/null | grep -v "scan_release_leaks" > /tmp/_leak_hits.$$ || true
  if [ -s /tmp/_leak_hits.$$ ]; then
    echo "⚠️  REVIEW — $name (allowed only inside disclaimers/negations):"
    head -10 /tmp/_leak_hits.$$
  fi
}

BLOCKED=0
echo "=== scanning $DIR ($(find "$DIR" -type f | wc -l | tr -d ' ') files) ==="

block "secret material" '(^|[^A-Za-z0-9])sk-[A-Za-z0-9]{10}|eyJ[A-Za-z0-9_-]{20}|BEGIN [A-Z ]*PRIVATE KEY|(^|[^A-Za-z0-9_])hf_[A-Za-z0-9]{20}|Bearer [A-Za-z0-9._-]{10}|api[_-]?key["'"'"' ]*[:=]["'"'"' ]*[A-Za-z0-9_/-]{12,}|password["'"'"' ]*[:=]["'"'"' ]*[A-Za-z0-9_/-]{8,}|ANTHROPIC_AUTH_TOKEN|security find-generic-password)'
block "personal identity" '(Raymond|Chau Yik|Yik-?Chun|周益俊|周奕俊|rayc00210|raymond\.thu|3033369600|seniordeli\.com|@pinnacle-robotics)'
block "related ventures (no cross-venture leaks)" '(SeniorDeli|seniordeli|康乐龄|康樂齡|kangleling|Carewells|華瓏|Pinnacle|pinnacle|Asaptic|asaptic|Getz|LinPig|linpig|PQSafe|Kinaite|kinaite|Shanyoule|膳友乐)'
block "internal infra" '(/Users/|/home/tun|OneDrive|CloudStorage|dgx-spark|100\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.|tailscale|\.ssh|crontab|mihomo|DMIT|hf-mirror)'
block "banned superlatives / competitor" '(唯一[^媒媒]|全港首創|首个|No\. ?1|幸福元氣)'

review "medical-adjacent wording" '(prevent|prevention|diagnos|treat(ment)?[^ ]*|clinical|clinically|防止|预防|預防|诊断|診斷|治疗|治療|临床|臨床|safe to eat|unsafe|aspirat|choking|呛咳|嗆咳|误吸|誤吸)'
review "first-claim wording" '(first IDDSI|首个IDDSI|首個IDDSI|first publicly)'

echo "=== scan done ==="
rm -f /tmp/_leak_hits.$$
if [ "$BLOCKED" = 1 ]; then echo "RESULT: 🚫 REFUSED — resolve BLOCK hits and re-run"; exit 1; fi
echo "RESULT: ✅ no BLOCK hits (eyeball the REVIEW lines)"
exit 0
