#!/usr/bin/env bash
# Every protocol on the six model pairs of Table 1 (GPT-4 split, temperature 0, two rounds).
set -euo pipefail

for m in mini l70b r1 ms26; do
    python -m colmad.run_single -m "$m"
done

for pair in mini:l70b l70b:mini mini:r1 l70b:r1 mini:ms26 l70b:ms26; do
    j1=${pair%%:*}; j2=${pair##*:}
    case "$pair" in
        mini:l70b|mini:ms26|l70b:ms26) quote=lowercase_problem ;;
        *) quote=insensitive ;;
    esac
    python -m colmad.run_colmad -j1 "$j1" -j2 "$j2" --quote_check "$quote"
    python -m colmad.run_copmad -j1 "$j1" -j2 "$j2"
    python -m colmad.run_mp     -j1 "$j1" -j2 "$j2"
    python -m colmad.run_cosmad -j1 "$j1" -j2 "$j2"
    echo "== $pair"
    python -m colmad.evaluate all -j1 "$j1" -j2 "$j2"
done
