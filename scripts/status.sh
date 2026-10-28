#!/usr/bin/env bash
echo "=== docker ==="
docker ps --format '{{.Names}} {{.Status}}'
echo "=== yosys log tail ==="
tail -4 /home/peng/flashattn_expmul/syn/d4_base/yosys.log 2>/dev/null
echo "=== synth dirs ==="
du -sh /home/peng/flashattn_expmul/syn/* 2>/dev/null
echo "=== fetch ==="
tail -3 /home/peng/flashattn_expmul/fetch.log
du -sh /home/peng/flashattn_expmul/models/flan-t5-base
ls -la /home/peng/flashattn_expmul/models/flan-t5-base | head -8
