#!/usr/bin/env bash
# Install Belka for the current user (no sudo): a private virtualenv, a
# launcher in ~/.local/bin and an entry in the GNOME app grid.
set -euo pipefail

APP_ID="io.github.romeruu_dev.Belka"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
VENV="$DATA/belka/venv"
BIN="$HOME/.local/bin"

if ! python3 -c "import venv, ensurepip" 2>/dev/null; then
    echo "Falta python3-venv:  sudo apt install python3-venv" >&2
    exit 1
fi

echo "→ Entorno de Python en $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$SRC"

echo "→ Lanzador en $BIN/belka"
mkdir -p "$BIN"
ln -sf "$VENV/bin/belka" "$BIN/belka"

echo "→ Icono y entrada de escritorio"
install -Dm644 "$SRC/belka/data/icons/belka.svg" "$DATA/icons/hicolor/scalable/apps/$APP_ID.svg"
mkdir -p "$DATA/applications"
sed "s|@EXEC@|$VENV/bin/belka|" "$SRC/packaging/$APP_ID.desktop.in" > "$DATA/applications/$APP_ID.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database -q "$DATA/applications" || true
# No icon-theme.cache of our own: a user-level cache goes stale and hides
# icons other apps add later. Refresh one only if it already existed, and
# bump the directory so GNOME Shell rescans it.
if [ -f "$DATA/icons/hicolor/icon-theme.cache" ] && command -v gtk-update-icon-cache >/dev/null; then
    gtk-update-icon-cache -q -t "$DATA/icons/hicolor" 2>/dev/null || true
fi
touch "$DATA/icons/hicolor"

# libgphoto2's udev rule gives cameras to the plugdev group.
if ! id -nG | tr ' ' '\n' | grep -qx plugdev; then
    echo "Aviso: tu usuario no está en el grupo plugdev; si la cámara no responde:"
    echo "  sudo usermod -aG plugdev $USER   (y vuelve a iniciar sesión)"
fi
echo "Listo. Abre «Belka» desde las aplicaciones o ejecuta: belka"
