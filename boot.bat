@echo off

:: Activate the virtual environment
call .\venv\Scripts\activate
::turning on the ollama servers
start cmd /k "echo Running the ollama model and embedder && ollama run llama3.2 && ollama run mxbai-embed-large"
::launching the webui server
start cmd /k "echo Running open-webui server && open-webui serve"
::launching the pipeline server
start "" "%PROGRAMFILES%\Git\bin\sh.exe" --login -c "echo Running pipelines server && ./start.sh"
