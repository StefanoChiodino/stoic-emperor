#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

TAILWIND_BIN="$PROJECT_ROOT/bin/tailwindcss"

if [ ! -f "$TAILWIND_BIN" ]; then
    echo "Downloading Tailwind CLI..."
    mkdir -p "$PROJECT_ROOT/bin"
    
    OS="$(uname -s)"
    ARCH="$(uname -m)"
    
    case "$OS-$ARCH" in
        Darwin-arm64) BINARY="tailwindcss-macos-arm64" ;;
        Darwin-x86_64) BINARY="tailwindcss-macos-x64" ;;
        Linux-x86_64) BINARY="tailwindcss-linux-x64" ;;
        Linux-aarch64) BINARY="tailwindcss-linux-arm64" ;;
        *) echo "Unsupported platform: $OS-$ARCH"; exit 1 ;;
    esac
    
    curl -sLo "$TAILWIND_BIN" "https://github.com/tailwindlabs/tailwindcss/releases/latest/download/$BINARY"
    chmod +x "$TAILWIND_BIN"
fi

if [ "$1" = "--watch" ]; then
    echo "Watching for changes..."
    "$TAILWIND_BIN" -i ./src/web/static/input.css -o ./src/web/static/output.css --watch
else
    echo "Building CSS..."
    "$TAILWIND_BIN" -i ./src/web/static/input.css -o ./src/web/static/output.css --minify
    echo "Done: src/web/static/output.css"
fi
