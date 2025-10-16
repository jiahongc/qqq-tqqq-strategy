## macOS launchd scheduling (09:40 ET)

1) Create a plist at `~/Library/LaunchAgents/com.qqq-tqqq.daily.plist`:

```
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.qqq-tqqq.daily</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/$(whoami)/Desktop/Coding/tqqq-backtesting/.venv/bin/python</string>
    <string>/Users/$(whoami)/Desktop/Coding/tqqq-backtesting/live/run_daily.py</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>9</integer>
    <key>Minute</key>
    <integer>40</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>/tmp/qqq-tqqq.out.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/qqq-tqqq.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
</dict>
</plist>
```

2) Load the job:

```bash
launchctl load ~/Library/LaunchAgents/com.qqq-tqqq.daily.plist
launchctl start com.qqq-tqqq.daily
```

3) Check logs:

```bash
tail -n 200 -f /tmp/qqq-tqqq.out.log /tmp/qqq-tqqq.err.log
```

4) Unload:

```bash
launchctl unload ~/Library/LaunchAgents/com.qqq-tqqq.daily.plist
```

Notes:
- Ensure the virtualenv path is correct.
- Ensure `.env` exists in project root with Alpaca creds.

