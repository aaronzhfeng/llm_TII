#!/bin/bash
# Monitor SlimPajama-627B Download Progress

echo "========================================================================="
echo "SlimPajama-627B Download Monitor"
echo "========================================================================="
echo ""
echo "Target: 895 GB"
echo "Location: /raid/zhf004/huggingface_cache/datasets"
echo ""
echo "Press Ctrl+C to exit"
echo "========================================================================="
echo ""

while true; do
    clear
    echo "========================================================================="
    echo "SlimPajama-627B Download Progress"
    echo "========================================================================="
    echo ""
    
    # Get current size
    CURRENT=$(du -sb /raid/zhf004/huggingface_cache/datasets 2>/dev/null | awk '{print $1}')
    CURRENT_GB=$(echo "scale=2; $CURRENT / 1024 / 1024 / 1024" | bc)
    TARGET_GB=895
    
    # Calculate percentage
    if [ ! -z "$CURRENT" ] && [ "$CURRENT" -gt 0 ]; then
        PERCENT=$(echo "scale=1; ($CURRENT_GB / $TARGET_GB) * 100" | bc)
        echo "Downloaded: ${CURRENT_GB} GB / ${TARGET_GB} GB (${PERCENT}%)"
        
        # Progress bar
        FILLED=$(echo "scale=0; $PERCENT / 2" | bc)
        printf "["
        for i in $(seq 1 50); do
            if [ $i -le ${FILLED:-0} ]; then
                printf "="
            else
                printf " "
            fi
        done
        printf "]\n"
    else
        echo "Downloaded: 0 GB / ${TARGET_GB} GB (0%)"
        echo "[                                                  ]"
    fi
    
    echo ""
    echo "Details:"
    du -sh /raid/zhf004/huggingface_cache/datasets/* 2>/dev/null | head -20
    
    echo ""
    echo "========================================================================="
    echo "Disk Space on /raid/:"
    df -h /raid/zhf004 | tail -n 1
    
    echo ""
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Refreshing every 10 seconds..."
    echo "========================================================================="
    
    sleep 10
done

