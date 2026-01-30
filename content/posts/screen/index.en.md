---
title: "Using `screen` to Manage Terminal Sessions"
date: 2025-05-21T22:15:00+08:00
# weight: 1
# aliases: ["/first"]
categories: ["Command Line Tools"]
tags: ["Server"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false # show table of contents
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
    image: "screen_cover.png" # image path/url
    alt: "Screen Cover Image" # alt text
    caption: "Cover image for the screen post" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## Basic `screen` commands

- Start a new `screen` session:

```bash
screen -S <name>
```

- List all `screen` sessions:

```bash
screen -ls
```

- Reattach a detached session:

```bash
screen -r <session-id>
```

- Detach (leave the session running in background):

Press `Ctrl+a`, then press `d`.

- Show the current session name (id.name):

```bash
echo $STY
```

- Close the current session:

Run `exit` (or press `Ctrl+d`) inside the session.

- Kill a session from outside:

```bash
screen -X -S <session_name> quit
```

- Clean up dead sessions:

```bash
screen -wipe
```
