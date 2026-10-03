#!/usr/bin/env bash
# Remove what install.sh added. Rolls in ~/Imágenes/Belka and preferences in
# ~/.config/belka are kept.
set -euo pipefail
APP_ID="io.github.romeruu_dev.Belka"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
rm -rf "$DATA/belka/venv"
rmdir "$DATA/belka" 2>/dev/null || true
rm -f "$HOME/.local/bin/belka" "$DATA/applications/$APP_ID.desktop" "$DATA/icons/hicolor/scalable/apps/$APP_ID.svg"
command -v update-desktop-database >/dev/null && update-desktop-database -q "$DATA/applications" || true
touch "$DATA/icons/hicolor" 2>/dev/null || true
echo "Belka desinstalado (tus rollos y preferencias siguen ahí)."
