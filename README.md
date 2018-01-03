# Lectures for MY459, LT 2018

## Course handout page: http://lse-my459.github.io

---

## How to submit homework

Handing in homework is basically committing changes you have made in your repository. There are 3 ways to do it:

1. use github desktop:
    1. click the changes tab in the left sidebar to see a list of the files that have been changed or added since the last commit.
Use the checkboxes to indicate which files should be part of the commit
    2. type your commit message in the Summary field and click the commit button to commit your changes
    3. click on sync on the top right corner of your github desktop app and it will sync to the online repository

    Reference:

     * https://stackoverflow.com/questions/27211578/how-to-commit-changes-to-github-with-github-desktop
      * https://services.github.com/on-demand/github-desktop/add-commits-github-desktop

2. use command line:
    1. cd [your github repository location]
    2. git commit -m "your comment"
    3. git push origin master

    Reference:
    * https://stackoverflow.com/questions/10364429/how-to-commit-to-remote-git-repository

3. other ways:

    a. update files on Github.com:

     * You can upload your updated files by clicking the "upload files" button (see here: https://help.github.com/articles/adding-a-file-to-a-repository/) to replace the outdated files (NOT recommended, you should be able to use the commit and push!)
     * if your files are in some simple format (like .md), you can edit the files directly

    b. email your homework to Ken (NOT recommended, use this ONLY IF you cannot use method 1 or 2)


**It is always a good idea to check if your commits are truly reflected on Github!**

---

## How to update the assignment starting codes / instructions

1. Clone your own assignment repository into your own computer if you haven't:
    ```
    git clone https://github.com/[your_user_name]/[your_assignment_repository]/
    ```
2. Change your current directory to the cloned repository if you haven't:
    ```
    cd [your_assignment_repository]
    ```
3. Fetch and merge changes from the latest version of the assignment instruction on the remote server to your working directory:
    ```
    git pull https://github.com/lse-my459/[assignment_instruction_repository]/
    ```
4. Resolve conflicts if there are any
5. Commit the changes back to your local repository:
    ```
    git commit -a -m "[your own comment]"
    ```
6. Push the merge to your GitHub repository:
    ```
    git push
    ```

Reference: https://help.github.com/articles/merging-an-upstream-repository-into-your-fork/
