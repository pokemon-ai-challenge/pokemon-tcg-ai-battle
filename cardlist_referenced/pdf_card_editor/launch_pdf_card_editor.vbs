Set shell = CreateObject("WScript.Shell")
scriptPath = CreateObject("Scripting.FileSystemObject").BuildPath(CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName), "launch_pdf_card_editor.ps1")
shell.Run "powershell.exe -ExecutionPolicy Bypass -File """ & scriptPath & """", 0, False
