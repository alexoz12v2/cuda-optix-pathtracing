# Current Directory Management 
(From Microsoft's documentation, but applies everywhere)
- Multithreaded applications and shared library code should avoid calling the SetCurrentDirectory function due to the risk of affecting relative path calculations being performed by other threads. Conversely,
- multithreaded applications and shared library code should avoid using relative paths so that they are unaffected by changes to the current directory performed by other threads.

Directory management, and, conversely, file path specification, is less trivial than it appears, due to the constraint of having to support multiple operating systems

## Windows
Reference: [Link](https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file)
In windows, a *Path* is composed of two sections
- *Disk Designator*, which can either be a drive letter followed by a column (eg `C:`), or server name followed by share name (eg `\\servername\sharename`) (network file system, path in this case is called *UNC Paths* (Universal Naming Conventions))
  *Volume Designators* (synonym) are case insensitive
  In case the disk designator is prepended by "\\.\" or "\\?\" (first for an object like pipes or mailboxes, second for long paths). Finally, instead of the drive letter, you can also use the Volume GUID
- *Directory*, following the disk designator. By default, these are *Case Insensitive*, but path (specifically the directory part) can be treated as case sensitive by a file 
  creation function if `FILE_FLAG_POSIX_SEMANTICS` si supplied (Win32 File Creation function, not NT ones)
  NTFS Supports POSIX Semantics but it's not the default Behaviour (Why?)
Here's an example in C# which illustrates all the different methods to address the same file (except for GUID)
```cs
using System;
using System.IO;

class Program
{
    static void Main()
    {
        string[] filenames = {
            @"c:\temp\test-file.txt",
            @"\\127.0.0.1\c$\temp\test-file.txt",
            @"\\LOCALHOST\c$\temp\test-file.txt",
            @"\\.\c:\temp\test-file.txt",
            @"\\?\c:\temp\test-file.txt",
            @"\\.\UNC\LOCALHOST\c$\temp\test-file.txt" };

        foreach (string filename in filenames)
        {
            FileInfo fi = new(filename);
            Console.WriteLine($"file {fi.Name}: {fi.Length:N0} bytes");
        }
    }
}
```

Each folder which makes up the *Directory* part of the *Path* is called *component*, and components are separated by a backslash `\`. A component or filename can contain any Unicode character except
- < (less than)
- > (greater than)
- : (colon)
- " (double quote)
- / (forward slash)
- \ (backslash)
- | (vertical bar or pipe)
- ? (question mark)
- * (asterisk)
- \0 (nul)
See documentation for a complete list
For each file you create, there are actually *2 File Names* associated with it (again, historical reasons, MS-DOS). *Long Name* and *Short Name*
- Long Name = arbitrary file name -> Retrieved by `GetLongPathName` (from the short name)
- Short Name = abbreviation of the Long name to adhere to the *8.3 fornat* -> Retrieved by `GetShortPathName` (from the long name)
- To retrieve a path from a File `HANDLE`, use `GetFinalPathNameByHandleW`

Examples:
| Long Name, posix semantics | short name (always case insensitive) |
|----------------------------|--------------------------------------|
| more.dots.txt              | MOREDO~1.TXT                         |
|----------------------------|--------------------------------------|

Note: the short path name will be given only if 
- The file is in an NTFS File System (or any other filesystem which supports it)
- 8.3 name generation enabled, and you can query it in powershell/cmd with (corresponds to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem\NtfsDisable8dot3NameCreation`)
  ```powershell
  fsutil behavior query disable8dot3
  ```
  - `0`: 8.3 name generation enabled for all volumes
  - `1`: 8.3 name generation disabled for all volumes
  - `2`: 8.3 name generation disabled for the current volume
in case in the current volume 8.3 name generation is disabled (as it should be as it is a thing of the past), then `GetShortPathNameW` will give you the long name
Example code with its output
```cpp

void printPaths(HANDLE hFile)
{
    wchar_t finalPath[MAX_PATH]{}; // ensure L`\0`termination 
    wchar_t finalPathNT[MAX_PATH]{};
    wchar_t shortPath[MAX_PATH]{};
    DWORD const finalPathLength   = GetFinalPathNameByHandleW(hFile, finalPath, MAX_PATH, FILE_NAME_OPENED | VOLUME_NAME_DOS);
    if (finalPathLength == 0)   printLastError(L"GetFinalPathNameByHandleW 1");
    DWORD const finalPathNTLength = GetFinalPathNameByHandleW(hFile, finalPathNT, MAX_PATH, FILE_NAME_OPENED | VOLUME_NAME_NT);
    if (finalPathNTLength == 0) printLastError(L"GetFinalPathNameByHandleW 2");
    DWORD const shortPathLength   = GetShortPathNameW(finalPath, shortPath, MAX_PATH);
    if (shortPathLength == 0)   printLastError(L"GetShortPathNameW");
    std::wstring_view const finalPathView{finalPath, finalPathLength << 1};
    std::wstring_view const finalPathNTView{finalPathNT, finalPathNTLength << 1};
    std::wstring_view const shortPathView{shortPath, shortPathLength << 1};
    std::wcout << L"Final Path (Drive Letter):     "   << finalPathView 
               << L"\nFinal Path (NT Device Object): " << finalPathNTView
               << L"\nShort Path:                    " << shortPathView
               << std::endl;
}

// output on my machine (disabled 8.3 name generation):
// Final Path (Drive Letter):     \\?\Y:\win32Experiments\more.dots.txt
// Final Path (NT Device Object): \Device\HarddiskVolume3\win32Experiments\more.dots.txt
// Short Path:                    \\?\Y:\win32Experiments\more.dots.txt
```

Notice the "\\?\" prefix on the path. This indicates that its a *Long Path*.
- *File Explorer Does not support long paths. If you try to create a directory and exceed the 260 limit, file explorer refuses to create a path*
  Therefore it is strongly advised to use a third party file explorer until Microsoft doesn't enable long path support for File Explorer, like
  - [Total Commander](https://www.ghisler.com/)
  - [Direcotry Opus](https://www.gpsoft.com.au/)
  It is also advisable to set your choice as the default file explorer by following [This Guide](https://superuser.com/questions/1031250/replace-windows-explorer-as-default-file-viewer-in-windows-10)
  And [This](https://support.microsoft.com/en-us/topic/how-to-back-up-and-restore-the-registry-in-windows-855140ad-e318-2a13-2829-d428a2ab0692) to create a backup beforehand

There is an important issue we need to consider: *Long Paths are not a default*. To create/open a file with a path larger than `MAX_PATH` (macro defined inside some windows header equal to 260 (nul terminator included))
you need to request for it explicitly with "\\?\", and following it must be an *Absolute Path*. There's more to it:
Whenever you start a process in windows, the first thing the OS will try to do is set the *Current Working directory* for the process, and windows expects it *NOT* to be a long path.
Trying to start an executable from a path longer than 260 will yield this funny error:
```sh
>> ./main.exe 
Error: Current working directory has a path longer than allowed for a
Win32 working directory.
Can't start native Windows application from here.
```
More Info on [This Link](https://stackoverflow.com/questions/63907045/starting-win32-application-from-a-long-path)
For a quick test, 
- go to Total Commander and create a path long enough to break the 260 limit. 
- Then, try to execute the following powershell command: `whoami` (which is on the `$Env:Path`). It will fail as the console cannot Set the current directory of the process

So we cannot execute anything which is inside a long path, unless you execute it through a symbolick link, example:
```powershell
PS any non long path>> New-Item -ItemType SymbolicLink -Path $LongPathFile -Value $Env:USERPROFILE
PS any non long path>> (Get-Item "$Env:USERPROFILE\temp_symlink.exe").Target # just to check
PS any non long path>> Invoke-Expression "$Env:USERPROFILE\temp_symlink.exe" # you still cannot execute this from a long path, so the cwd is skrewed
PS any non long path>> Remove-Item -Path "$Env:USERPROFILE\temp_symlink.exe"
```

What about programmatic process creation through `CreateProcessW`? Does that handle long paths? Yes, only if you supply the module name through the `lpApplicationName` and NOT through `lpCommandLine` ([Reference](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw))
```cpp
    PROCESS_INFORMATION procInfo{};
    STARTUPINFOW info{};
    info.cb = sizeof(STARTUPINFOW);

    wchar_t const* longExe = L"\\\\?\\Y:\\win32Experiments\\very long -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\\inside long folder-----------------------\\main.exe";
    bool success = CreateProcessW(longExe, nullptr, nullptr, nullptr, false, 0, nullptr, nullptr, &info, &procInfo);
    if (!success)
        std::wcerr << L"Why" << std::endl;
    else
        std::wcout << L"Nice" << std::endl;
```
Apparently, this does work, so the problem lies within the windows console.
- Do `GetCurrentDirectoryW` and `SetCurrentDirectoryW` work when the application is `longPathsAware`? Yes, only if you supply an absolute path directly with the "\\?\" prefix to the set function
So to 
- open a file safely, we need to construct its absolute path (or, even better, be supplied an absolute path to begin with), and construct an explicit long path
- create a process safely, we supply a long path to `lpApplicationName`, and don't set its start current working directory over the 260 limit (the child should handle it manually with `SetCurrentDirectoryW`)

## How to handle Long Paths then 
https://stackoverflow.com/questions/38036943/getfullpathnamew-and-long-windows-file-paths

## File Mapping
https://learn.microsoft.com/en-us/windows/win32/memory/file-mapping