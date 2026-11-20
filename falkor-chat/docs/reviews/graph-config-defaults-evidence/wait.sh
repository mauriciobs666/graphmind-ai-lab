until ! docker ps --format '{{.Names}}' | grep -q k098-gdba-tc; do sleep 5; done
