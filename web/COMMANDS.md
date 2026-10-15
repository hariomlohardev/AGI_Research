# Global Command Prompt shortcut

Installed at `C:\Users\hario\bin\cs.cmd`, in an existing user PATH directory.

- `cs`: changes the current Command Prompt directory (including drive) to `D:\Claude code\curriculum-system` and opens that folder in a new File Explorer window.
- `cs --web`: changes the current Command Prompt to that directory, starts the local dashboard, and opens the default browser. Press **Ctrl+C** (confirm the batch termination if Windows asks) or close that Command Prompt to stop the server.
- `cs --help`: shows usage.

There is no login startup entry, scheduled task, or background service. Running `cs` alone does not start the web server. If `cs --web` already started a dashboard in another Command Prompt, a second call opens that dashboard instead of launching a duplicate.

These shortcuts are for **Command Prompt (cmd.exe)**. PowerShell can launch the web shortcut, but a `.cmd` file cannot change the parent PowerShell working directory. When calling from another batch script, use `call cs`.

The web launcher is `web/scripts/cs-web.mjs`. To move the repository, update the two hardcoded directory paths and launcher path in `cs.cmd`. All curriculum data is read-only.
