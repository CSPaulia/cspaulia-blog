---
title: "PaperMod进化论"
date: 2025-06-07T17:32:03+08:00
# weight: 1
# aliases: ["/first"]
draft: true
categories: ["建站"]
tags: ["Hugo", "PaperMod", "建站"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: ""
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: true
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
    relative: true # when using page bundles set this to true
    hidden: true # only hide on current single page
    hiddenInList: true # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## 自定义post_meta显示

- **最后编辑(Lastmod)**

    在`config.yaml`中添加：

    ```yaml
    frontmatter:
        date:
            - date
            - publishDate
            - lastmod
        lastmod:
            - :git
            - :fileModTime
            - lastmod
            - date
            - publishDate
    ```

    最后编辑时间會根據frontmatter中的順序取值
    - `:git`：會去抓git提交紀錄的日期，且必須於config.yml中啟用enableGitInfo = true(沒試成功)
    - `:fileModTime`：根據本機的文件最後修改紀錄
    - `lastmod`：可以在文章的frontmatter區塊中直接設定
    - `date`：可以在文章的frontmatter區塊中直接設定
    - `publishDate`：文章發布的日期

- **修改 post_meta.html 文件**

    ```html
    <!-- layouts/partials/post_meta.html -->
    {{ $scratch := newScratch }}

    {{ if not .Date.IsZero }}
    {{ $scratch.Add "meta" (slice (printf `<span class="post-meta-item"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg> %s</span>` (.Date | time.Format (default "January 2, 2006" site.Params.DateFormat)))) }}
    {{ end }}

    {{ if not .Lastmod.IsZero }}
    {{ $scratch.Add "meta" (slice (printf `<span class="post-meta-item"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19l7-7 3 3-7 7-3-3z"></path><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"></path><path d="M2 2l7.586 7.586"></path><circle cx="11" cy="11" r="2"></circle></svg> %s</span>` (.Lastmod | time.Format (default "January 2, 2006" site.Params.DateFormat)))) }}
    {{ end }}

    {{ if (.Param "ShowWordCount") }}
    {{ $scratch.Add "meta" (slice (printf `<span class="post-meta-item"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><path d="M14 2v6h6"></path><line x1="8" y1="13" x2="16" y2="13"></line><line x1="8" y1="17" x2="16" y2="17"></line><line x1="8" y1="9" x2="12" y2="9"></line></svg> %s</span>` (i18n "words" .WordCount | default (printf "%d words" .WordCount)))) }}
    {{ end }}

    {{ if (.Param "ShowReadingTime") }}
    {{ $scratch.Add "meta" (slice (printf `<span class="post-meta-item"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg> %s</span>` (i18n "read_time" .ReadingTime | default (printf "%d min" .ReadingTime)))) }}
    {{ end }}

    {{ with (partial "author.html" .) }}
    {{ $scratch.Add "meta" (slice (printf `<span class="post-meta-item"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg> %s</span>` .)) }}
    {{ end }}

    {{ with ($scratch.Get "meta") }}
    {{ delimit . " ｜ " | safeHTML }}
    {{ end }}
    ```

- **添加 CSS 样式来调整图标的显示效果**

    ```css
    /* assests/css/extended/blank.css */
    .post-meta-item {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    .post-meta-item svg {
        stroke: var(--secondary);
    }
    ```

- **列表页面隐藏** （可选）

    若不希望列表页出现`ReadingTime`，则复制一份`layouts/partials/post_meta.html`，命名为`post_meta_list.html`，将`post_meta_list.html`中的如下代码删去

    ```html
    <!-- layouts/partials/post_meta_list.html -->
    <!-- 删掉 -->
    {{ if (.Param "ShowReadingTime") }}
        ...
    {{ end }}
    ```

    并在`layouts/_default/list.html`中将：

    ```html
    <!-- layouts/_default/list.html -->
    {{- partial "post_meta.html" . -}}
    ```

    修改为

    ```html
    <!-- layouts/_default/list.html -->
    {{- partial "post_meta_list.html" . -}}
    ```

## 代码块改进

### 代码块缩进设置

在 `blank.css` 中添加以下代码：

```css
/* assets/css/extended/blank.css */
.post-content pre { /* 代码块缩进样式 */
    margin-left: 2em;  /* 缩进距离 */
}
.post-content li pre {  /* 列表中的代码块额外缩进 */
    margin-left: 4em;  /* 列表中的代码块缩进更多 */
}
```

### PaperMod 主题的语法高亮设置

`config.yaml`中设置正确的语法高亮：

```yaml
params:
# ...existing code...
assets:
    disableHLJS: false  # 启用 highlight.js

# 设置代码高亮主题
syntax_highlighter: "highlight.js"
```

## 博客文章封面图片缩小并移到侧边

- **复制`list.html`**
    
    从`themes/PaperMod/layouts/_default/list.html`中复制一份`list.html`放置于`layouts/_default/list.html`，并将

    ```html
    <!-- layouts/_default/list.html -->
    <article class="{{ $class }}">
    {{- $isHidden := (.Param "cover.hiddenInList") | default (.Param "cover.hidden") | default false }}
    {{- partial "cover.html" (dict "cxt" . "IsSingle" false "isHidden" $isHidden) }}
    <header class="entry-header">
        <h2 class="entry-hint-parent">
        {{- .Title }}
        {{- if .Draft }}
        <span class="entry-hint" title="Draft">
            <svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 -960 960 960" fill="currentColor">
            <path
                d="M160-410v-60h300v60H160Zm0-165v-60h470v60H160Zm0-165v-60h470v60H160Zm360 580v-123l221-220q9-9 20-13t22-4q12 0 23 4.5t20 13.5l37 37q9 9 13 20t4 22q0 11-4.5 22.5T862.09-380L643-160H520Zm300-263-37-37 37 37ZM580-220h38l121-122-18-19-19-18-122 121v38Zm141-141-19-18 37 37-18-19Z" />
            </svg>
        </span>
        {{- end }}
        </h2>
    </header>
    {{- if (ne (.Param "hideSummary") true) }}
    <div class="entry-content">
        <p>{{ .Summary | plainify | htmlUnescape }}{{ if .Truncated }}...{{ end }}</p>
    </div>
    {{- end }}
    {{- if not (.Param "hideMeta") }}
    <footer class="entry-footer">
        {{- partial "post_meta.html" . -}}
    </footer>
    {{- end }}
    <a class="entry-link" aria-label="post link to {{ .Title | plainify }}" href="{{ .Permalink }}"></a>
    </article>
    ```

    修改为

    ```html
    <!-- layouts/_default/list.html -->
    <article class="{{ $class }}">
        <div class="post-info">
            <header class="entry-header">
                <h2>{{ .Title }}</h2>
            </header>
            {{- if .Description }}
            <section class="entry-content">
                <p>{{ .Description }}</p>
            </section>
            {{- else if (ne (.Param "hideSummary") true) }}
            <section class="entry-content">
                <p>{{ .Summary | plainify | htmlUnescape }}{{ if .Truncated }}...{{ end }}</p>
            </section>
            {{- end }}
            {{- if not (.Param "hideMeta") }}
            <footer class="entry-footer">
                {{- partial "post_meta.html" . -}}
            </footer>
            {{- end }}
        </div>
        {{- $isHidden := (.Param "cover.hiddenInList") | default (.Param "cover.hidden") | default false }}
        {{- partial "cover.html" (dict "cxt" . "IsHome" true "isHidden" $isHidden) }}
        <a class="entry-link" aria-label="post link to {{ .Title | plainify }}" href="{{ .Permalink }}"></a>
    </article>
    ```

- **添加自定义样式**
    
    ```css
    /* assets/css/extended/blank.css */
    .post-entry {
        display: flex;
        flex-direction: row;
        align-items: center;
    }

    .entry-cover {
        overflow: hidden;
        padding-left: 18px;
        height: 100%;
        width: 50%;
        margin-bottom: unset;
    }

    .post-info {
        display: inline-block;
        overflow: hidden;
        width: 90%;
    }
    ```

- **侧边首图放大动画**

    ```css
    /* assets/css/extended/blank.css */
    .post-entry img{
        transition: all 0.3s ease-out;
        transform:scale(1,1);
    }
    .post-entry:hover img{
        transition: all 0.3s ease-out;
        transform:scale(1.02,1.02);
    }
    ```