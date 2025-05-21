---
title: "screen基本命令"
date: 2020-09-15T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["first"]
author: "Me"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Desc Text."
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/<path_to_repo>/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

# screen基本命令
- 新建一个screen会话
```
screen -S <名字>
```

- 查看所有screen会话
```
screen -ls
```

- 恢复之前分离的会话
```
screen -r <会话ID>
```

- 退出当前screen会话
```
键盘点击ctrl+a , 然后按d
```

- 查看当前所在会话(id.name)
```
echo $STY
```

- 关闭会话
如果在会话之中，输入exit或者Ctrl+d来终止这个会话。成功终止后，如果有其他处于Attached状态的screen界面，他就会跳到那个界面中，如果没有，他就会跳到默认界面上。

- 删除会话
```
screen -X -S session_name quit
```

- 清理会话
```
screen -wipe #清理那些dead的会话
```
