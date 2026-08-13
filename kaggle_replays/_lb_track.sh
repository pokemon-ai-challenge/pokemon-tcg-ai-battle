#!/usr/bin/env bash
# LBスコアの推移を記録して収束を見る。
# publicScore は提出直後から動き続ける(実測で10点規模)。1回の値で優劣を判断しないための記録。
# 20分ごとにサンプリングし、前回から変化した提出だけを1行で出す。
set -u
cd "$(dirname "$0")"
OUT=_lb_history.tsv
[ -f "$OUT" ] || printf "timestamp\tref\tscore\n" > "$OUT"

declare -A last
while :; do
  ts=$(date -u +%FT%TZ)
  PYTHONUTF8=1 kaggle competitions submissions -c pokemon-tcg-ai-battle --format json 2>/dev/null | python -c "
import sys,json
s=sys.stdin.read()
i=min((x for x in (s.find('['),s.find('{')) if x!=-1), default=-1)
if i<0: raise SystemExit
for r in json.JSONDecoder().raw_decode(s[i:])[0][:6]:
    sc=r.get('publicScore')
    if sc: print('%s\t%s' % (r.get('ref'), sc))
" > /tmp/_lb_now.tsv 2>/dev/null || true

  while IFS=$'\t' read -r ref score; do
    [ -z "${ref:-}" ] && continue
    prev=${last[$ref]:-}
    if [ "$prev" != "$score" ]; then
      printf "%s\t%s\t%s\n" "$ts" "$ref" "$score" >> "$OUT"
      if [ -n "$prev" ]; then echo "$ts  ref=$ref  $prev -> $score"; else echo "$ts  ref=$ref  $score (初回記録)"; fi
      last[$ref]=$score
    fi
  done < /tmp/_lb_now.tsv
  sleep 1200
done
