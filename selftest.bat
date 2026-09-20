@echo off

setlocal

cd /d "%~dp0"

rem 自检 runner。缺哪个环境就 SKIP 哪项，不会替你装依赖。

".venv\Scripts\python.exe" -X utf8 -m tools.selftest %*

set "CODE=%ERRORLEVEL%"

echo.

pause

exit /b %CODE%
