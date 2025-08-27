[Setup]
AppName=Whisper Transcriber
AppVersion=1.0
DefaultDirName={pf}\\WhisperTranscriber
DefaultGroupName=Whisper Transcriber
OutputDir=installer
OutputBaseFilename=WhisperTranscriberSetup
Compression=lzma
SolidCompression=yes
DisableProgramGroupPage=yes

[Files]
Source: "dist\\WhisperTranscriber.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "ffmpeg\\ffmpeg.exe"; DestDir: "{app}\\ffmpeg"; Flags: ignoreversion
Source: "belle-whisper-large-v3-turbo-ct2f16\\*"; DestDir: "{app}\\belle-whisper-large-v3-turbo-ct2f16"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\\Whisper Transcriber"; Filename: "{app}\\WhisperTranscriber.exe"
Name: "{userdesktop}\\Whisper Transcriber"; Filename: "{app}\\WhisperTranscriber.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked
