---
title: "如何上传本地文件到Github库"
date: 2025-05-21T22:07:00+09:00
# weight: 1
# aliases: ["/first"]
categories: ["Tips"]
tags: ["Github"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: ""
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
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
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## 上传本地文件到Github库

### 1. 在GitHub上创建仓库（远程仓库）

在 GitHub 上创建一个新的代码仓库

### 2. 安装/配置Git
1. 安装略，配置 Git：
        
        git config --global user.name "Your Name"
        git config --global user.email "your_email@example.com"

这里的 "Your Name" 和 "your_email@example.com" 分别为你的用户名和邮箱地址

3. 检查 Git 是否已经配置用户名和邮箱：

        git config --global user.name
        git config --global user.email


### 3. 上传文件到Github（本地仓库–>远程仓库）
1. 初始化 Git 仓库, 执行以下命令：
    
        git init

2. 将 csj_project 项目文件夹添加到本地仓库中，执行以下命令：
    
        git add csj_project/
        #或者输入
        git add .

3. 将当前工作目录中的更改保存到本地代码仓库中，执行以下命令：
    
        git commit -m "Initial commit"

4. 在 GitHub 上创建一个新的远程仓库，获取复制该仓库的 SSH 或 HTTPS 链接

5. 将本地仓库与远程仓库进行关联，执行以下命令：
    
        git remote add origin <远程仓库链接> 
        <远程仓库链接>就是你刚才复制的仓库的 SSH 或 HTTPS 链接，例如我的就是： 
        https://github.com/CSPaulia/(库的名称).git 
        完整命令 
        git remote add origin https://github.com/CSPaulia/(库的名称).git

6. 将本地仓库中的代码推送到远程仓库中，执行以下命令：
    
        #git push origin 分支名
        #完整命令
        git push -u origin master
