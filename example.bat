if exist "GeneratedCode" rd /q /s "GeneratedCode"
mkdir GeneratedCode
python generate_example.py
copy project.csproj GeneratedCode\
copy *.cs GeneratedCode\
copy test_dog.png GeneratedCode\
@echo off
for /r c:\Windows\Microsoft.NET\Framework %%a in (*) do if "%%~nxa"=="MSBuild.exe" set p=%%~dpnxa
if defined p (
echo MSBUILD FOUND!!!
) else (
echo MSBUILD NOT FOUND!!!
)
@echo on

%p% GeneratedCode\project.csproj /p:Configuration=Release /t:Build


cd GeneratedCode
MyModel.exe
