#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="AttentionMonitor"
APP_BUNDLE="${ROOT_DIR}/dist/${APP_NAME}.app"
INFO_PLIST="${APP_BUNDLE}/Contents/Info.plist"

cd "${ROOT_DIR}"

if [[ ! -d "${ROOT_DIR}/venv" ]]; then
  echo "ERROR: venv not found. Create it and install dependencies first."
  exit 1
fi

echo "==> Activating venv"
source "${ROOT_DIR}/venv/bin/activate"

echo "==> Cleaning previous build"
rm -rf "${ROOT_DIR}/build" "${ROOT_DIR}/dist"

echo "==> Building app with PyInstaller spec"
pyinstaller --noconfirm "${ROOT_DIR}/AttentionMonitor.spec"

if [[ ! -f "${INFO_PLIST}" ]]; then
  echo "ERROR: Info.plist not found at ${INFO_PLIST}"
  exit 1
fi

echo "==> Adding camera usage description"
/usr/libexec/PlistBuddy -c "Add :NSCameraUsageDescription string \"需要使用摄像头进行注意力监控\"" "${INFO_PLIST}" 2>/dev/null || \
/usr/libexec/PlistBuddy -c "Set :NSCameraUsageDescription \"需要使用摄像头进行注意力监控\"" "${INFO_PLIST}"

echo "==> Clearing quarantine (Finder double-click compatibility)"
xattr -dr com.apple.quarantine "${APP_BUNDLE}" || true

echo "==> Ad-hoc code signing"
codesign --force --deep --sign - "${APP_BUNDLE}"

echo "==> Done: ${APP_BUNDLE}"
