# Git Submodules 
To add another repository as part of your own repository, without necessarely storing physically its files all the time and track them with `git`, we can use `git submodule`.
Adding a repository with the command
```
git submodule add https://github.com/mmp/pbrt-v4-scenes.git ./res/scenes
```

This command won't actually add the files contained in the target repository, (actually, newer versions of git should) but it will 
create/edit a file called `.gitmodules`, which stores a map of (Repository URL, Directory) pair.

The directory which will contain the git submodule is created, *empty*. To fill it with its supposed content, 2 commands are necessary
1. Initialize local configuration file
   ```
   git submodule init
   ```
2. Fetch all the data from the submodule projects (this will do the cloning)
   ```
   git submodule update
   ```
Repositories used as submodules can actually contain submodules inside them too, and therefore you have the command which performs the two steps above and resurses through all
submodules
```
git submodule update --init --recursive
```

When cloning a repository with submodules, you can fetch submodules in one shot
```
git clone --recursive <URL>
```

[This Link](https://git-scm.com/book/en/v2/Git-Tools-Submodules) contains more info, eg. how we can add commits to a submodule

To remove a submodule from the current repository, run the command
```
git rm <path-to-submodule>
```
and commit.
There will still be records left about the presence of the submodule in the repository, to make it possible to backtrack to previous commit, eg. in `.git/modules/` and `.git/config`
If you want to remove this info too, run
```
rm -rf .git/modules/<path-to-submodule>
git config --remove-section submodule.<path-to-submodule>
```