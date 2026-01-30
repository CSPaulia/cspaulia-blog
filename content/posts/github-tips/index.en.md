---
title: "How to Use Git to Push Local Files to a GitHub Repository"
date: 2025-05-21T22:07:00+09:00
# weight: 1
# aliases: ["/first"]
categories: ["Command Line Tools"]
tags: ["Git"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: ""
# canonicalURL: "https://canonical.url/to/page"
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "git_github.png" # image path/url
    alt: "Git and GitHub" # alt text
    caption: "How to use Git to push local files to a GitHub repository" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## Push local files to GitHub

### 1. Create a repository on GitHub (remote)

Create a new repository on GitHub.

### 2. Install / configure Git

Configure your username and email:

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

Verify the current config:

```bash
git config --global user.name
git config --global user.email
```

### 3. Push files (local -> remote)

1. Initialize a Git repo:

```bash
git init
```

2. Add files:

```bash
# add a folder
git add csj_project/

# or add everything
git add .
```

3. Commit:

```bash
git commit -m "Initial commit"
```

4. Copy the repository URL (SSH or HTTPS) from GitHub.

5. Add the remote:

```bash
git remote add origin <repo-url>
```

Example:

```bash
git remote add origin https://github.com/CSPaulia/<repo-name>.git
```

6. Push to GitHub:

```bash
# if your default branch is master
git push -u origin master

# if your default branch is main
# git push -u origin main
```
