#!/bin/zsh
cd "$(dirname "$0")"
run() {
  echo "===== $1: $2"
  shift 2
  "$@" > .out 2> .err
  code=$?
  echo "--- stdout"; cat .out
  echo "--- stderr"; grep -v -i "warning" .err
  echo "--- код завершення: $code"
}
{
  run "unit" "функціональні тести сценаріїв" python -m unittest discover -s tests -v
  run "scenarios" "таблиця тестових прикладів" python -m forecast.cli scenarios
  run "holdout" "показники якості на відкладених даних" python -m forecast.cli evaluate
  run "json" "контракт JSON для Decision Engine" python -m forecast.cli forecast --store Київ-017 --sku MILK-1L --date 2026-10-24 --json
  run "bad-sku" "невідомий товар" python -m forecast.cli forecast --store Київ-017 --sku NO-SUCH --date 2026-10-24
} > tests_log.txt
rm -f .out .err
grep -c "код завершення: 0" tests_log.txt
