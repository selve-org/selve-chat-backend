#!/bin/bash
# Local check: Verify chat backend schema matches main backend schema
# Run this before committing changes to prisma/schema.prisma

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_SCHEMA="$SCRIPT_DIR/../prisma/schema.prisma"
MAIN_SCHEMA="$SCRIPT_DIR/../../selve/backend/schema.prisma"

echo "🔍 Checking schema sync..."

# Check if files exist
if [ ! -f "$CHAT_SCHEMA" ]; then
    echo "❌ Chat backend schema not found at $CHAT_SCHEMA"
    exit 1
fi

if [ ! -f "$MAIN_SCHEMA" ]; then
    echo "❌ Main backend schema not found at $MAIN_SCHEMA"
    exit 1
fi

# Compare file contents
if diff -q "$CHAT_SCHEMA" "$MAIN_SCHEMA" > /dev/null 2>&1; then
    echo "✅ Schemas are in sync!"
    exit 0
else
    echo "❌ SCHEMA MISMATCH!"
    echo "Chat backend: $CHAT_SCHEMA"
    echo "Main backend: $MAIN_SCHEMA"
    echo ""
    echo "Differences:"
    diff "$CHAT_SCHEMA" "$MAIN_SCHEMA" || true
    echo ""
    echo "⚠️  Both backends must use the same schema."
    echo "💡 Copy the main backend schema to chat backend:"
    echo "    cp $MAIN_SCHEMA $CHAT_SCHEMA"
    exit 1
fi
