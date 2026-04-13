Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c """ & Replace(WScript.ScriptFullName, ".vbs", ".bat") & """", 0, False
