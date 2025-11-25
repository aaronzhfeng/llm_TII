#!/bin/bash
# HuggingFace Pro Account Authentication Helper

echo "========================================================================="
echo "HuggingFace Pro Account Authentication"
echo "========================================================================="
echo ""
echo "Steps:"
echo "1. Go to: https://huggingface.co/settings/tokens"
echo "2. Create a new token (or copy existing one)"
echo "   - Name: 'SlimPajama Download'"
echo "   - Type: 'Read'"
echo "3. Copy the token"
echo "4. Paste it below when prompted"
echo ""
echo "========================================================================="
echo ""

# Activate venv
source /home/zhf004/llm_TII/venv/bin/activate

# Use new command
hf auth login

echo ""
echo "========================================================================="
echo "Verification:"
echo "========================================================================="
hf auth whoami

echo ""
echo "If you see your username above, authentication succeeded!"
echo "You can now run: python3 prepare.py --download_only"

