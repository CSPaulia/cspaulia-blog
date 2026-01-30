---
title: "The Evolution of PaperMod"
date: 2025-06-07T17:32:03+08:00
# weight: 1
# aliases: ["/first"]
categories: ["Web Development"]
tags: ["Hugo", "PaperMod", "Website"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "My PaperMod customization notes"
# canonicalURL: "https://canonical.url/to/page"
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
    image: "papermod.png" # image path/url
    alt: "Advanced PaperMod theme customization" # alt text
    caption: "Advanced PaperMod" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: true # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## Customize `post_meta` display

- **Last edited time (`Lastmod`)**

    Add this to `config.yaml`:

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

    The last modified time is resolved in the order listed in `frontmatter.lastmod`:
    - `:git`: uses git commit time; requires `enableGitInfo = true` in `config.yml` (I didn’t get this working)
    - `:fileModTime`: uses the local file’s last modified time
    - `lastmod`: set directly in the page front matter
    - `date`: set directly in the page front matter
    - `publishDate`: the publish time

- **Edit `post_meta.html`**

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

- **Add CSS to align icons**

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

- **Hide on list pages (optional)**

    If you don’t want `ReadingTime` to show on list pages, copy `layouts/partials/post_meta.html` to `layouts/partials/post_meta_list.html` and remove the following block from `post_meta_list.html`:

    ```html
    <!-- layouts/partials/post_meta_list.html -->
    <!-- remove -->
    {{ if (.Param "ShowReadingTime") }}
        ...
    {{ end }}
    ```

    Then in `layouts/_default/list.html`, change:

    ```html
    <!-- layouts/_default/list.html -->
    {{- partial "post_meta.html" . -}}
    ```

    to:

    ```html
    <!-- layouts/_default/list.html -->
    {{- partial "post_meta_list.html" . -}}
    ```

## Code block improvements

### Code block indentation

Add the following to `blank.css`:

```css
/* assets/css/extended/blank.css */
.post-content pre { /* code block indentation */
    margin-left: 2em;  /* indentation */
}
.post-content li pre {  /* extra indentation for code blocks inside lists */
    margin-left: 4em;
}
```

### Syntax highlighting settings in PaperMod

- Configure syntax highlighting correctly in `config.yaml`:

    ```yaml
    params:
    # ...existing code...
    assets:
        disableHLJS: false  # enable highlight.js

    # Set syntax highlighting engine
    syntax_highlighter: "highlight.js"
    ```

- Example fenced code block with highlighted lines:

    ```python {hl_lines=[3]}
    print("Line 1")
    print("Line 2")
    print("**This line will be highlighted**")
    print("Line 4")
    ```

### Limit code block height

- **Limit height and enable scrolling**

    ```css
    /* assets/css/extended/blank.css */
    .post-content pre {
        max-height: 400px;      /* max height */
        overflow: auto;         /* scroll when overflow */
    }
    ```

- **Scrollbar size**

    ```css
    .post-content pre::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ```

## Shrink post cover images and move them to the side

- **Copy `list.html`**

    Copy `themes/PaperMod/layouts/_default/list.html` to `layouts/_default/list.html`, and replace:

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

    with:

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

- **Add custom styles**

    ```css
    /* assets/css/extended/blank.css */
    .post-entry {
        display: flex;
        flex-direction: row;
        align-items: center;
    }
    .entry-cover {
        overflow: hidden;
        /* padding-left: 18px; */
        height: 100%;
        width: 50%;
        margin-bottom: unset;
        border-radius: 12px;
    }
    .entry-cover img {
        border-radius: 12px;
    }
    .post-info {
        display: inline-block;
        overflow: hidden;
        width: 90%;
    }
    ```

- **Hover zoom animation for side cover**

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

## Custom Post Footer

- **Copy `single.html`**

    Find `single.html` under `themes/PaperMod/layouts/_default` and copy it to `layouts/_default/single.html`.

- **Modify the `post-footer` section**

Find:

```html
<!-- layouts/_default/single.html -->
<footer class="post-footer">
    ...
</footer>
```

Replace it with:

```html
<!-- layouts/_default/single.html -->
<footer class="post-footer">
  {{- $tags := .Language.Params.Taxonomies.tag | default "tags" }}
  <p style="font-size: medium; margin-bottom: 5px; font-weight: bold;">Tags:</p>
  <ul class="post-tags">
    {{- range ($.GetTerms $tags) }}
    <li><a href="{{ .Permalink }}">{{ .LinkTitle }}</a></li>
    {{- end }}
  </ul>
  {{- $categories := .Language.Params.Taxonomies.categories | default "categories" }}
  <p style="font-size: medium; margin-bottom: 5px; font-weight: bold;">Categories:</p>
  <ul class="post-tags">
    {{- range ($.GetTerms $categories) }}
    <li><a href="{{ .Permalink }}">{{ .LinkTitle }}</a></li>
    {{- end }}
  </ul>
  {{- if (.Param "ShowPostNavLinks") }}
  {{- partial "post_nav_links.html" . }}
  {{- end }}
  {{- if (and site.Params.ShowShareButtons (ne .Params.disableShare true)) }}
  {{- partial "share_icons.html" . -}}
  {{- end }}
</footer>
```

- **Result**

<p align="center">
    {{< img src="post_footer.png" alt="post_footer" width="80%" >}}
</p>

## Add reference links at the end of posts

- Add a bit of style in `blank.css`

    ```css
    /* assets/css/extended/blank.css */
    /* light mode */
    .zhihu-ref {
        background: #f6f7fa;
        border-radius: 8px;
        padding: 1.2em 1.5em 1.2em 1.5em;
        margin-top: 2em;
        font-size: 1em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        border-left: 4px solid #0084ff;
        transition: background 0.3s, border-color 0.3s;
    }
    .zhihu-ref-title {
        font-weight: bold;
        color: #175199;
        margin-bottom: 0.5em;
        font-size: 1.1em;
    }
    .zhihu-ref a {
        color: #175199;
    }
    .zhihu-ref a:hover {
        color: #0084ff;
    }
    /* dark mode (PaperMod uses .dark) */
    .dark .zhihu-ref {
        background: #23272e;
        border-left: 4px solid #3ea6ff;
    }
    .dark .zhihu-ref-title,
    .dark .zhihu-ref a {
        color: #3ea6ff;
    }
    .dark .zhihu-ref a:hover {
        color: #8cc8ff;
    }
    ```

- Markdown usage:

    ```md
    <hr>
    <div class="references">
    <h3>References</h3>
    <ol>
        <li><a href="https://gohugo.io/documentation/" target="_blank">Hugo Documentation</a></li>
        <li><a href="https://github.com/adityatelange/hugo-PaperMod" target="_blank">PaperMod Theme</a></li>
        <li><a href="https://www.markdownguide.org/" target="_blank">Markdown Guide</a></li>
    </ol>
    </div>
    ```

- Result:

<p align="center">
    {{< img src="zhihu-ref.png" alt="zhihu-ref" width="80%" >}}
</p>

## Side Table of Contents (TOC)

- Create `toc.html` under `layouts/partials` and add:

    ```html
    <!-- layouts/partials/toc.html -->
    {{- $headers := findRE "<h[1-6].*?>(.|\n])+?</h[1-6]>" .Content -}}
    {{- $has_headers := ge (len $headers) 1 -}}
    {{- if $has_headers -}}
    <aside id="toc-container" class="toc-container wide">
        <div class="toc">
            <details {{if (.Param "TocOpen") }} open{{ end }}>
                <summary accesskey="c" title="(Alt + C)">
                    <span class="details">{{- i18n "toc" | default "Table of Contents" }}</span>
                </summary>

                <div class="inner">
                    {{- $largest := 6 -}}
                    {{- range $headers -}}
                    {{- $headerLevel := index (findRE "[1-6]" . 1) 0 -}}
                    {{- $headerLevel := len (seq $headerLevel) -}}
                    {{- if lt $headerLevel $largest -}}
                    {{- $largest = $headerLevel -}}
                    {{- end -}}
                    {{- end -}}

                    {{- $firstHeaderLevel := len (seq (index (findRE "[1-6]" (index $headers 0) 1) 0)) -}}

                    {{- $.Scratch.Set "bareul" slice -}}
                    <ul>
                        {{- range seq (sub $firstHeaderLevel $largest) -}}
                        <ul>
                            {{- $.Scratch.Add "bareul" (sub (add $largest .) 1) -}}
                            {{- end -}}
                            {{- range $i, $header := $headers -}}
                            {{- $headerLevel := index (findRE "[1-6]" . 1) 0 -}}
                            {{- $headerLevel := len (seq $headerLevel) -}}

                            {{/* get id="xyz" */}}
                            {{- $id := index (findRE "(id=\"(.*?)\")" $header 9) 0 }}

                            {{- /* strip id=\"\" to leave xyz, no way to get regex capturing groups in hugo */ -}}
                            {{- $cleanedID := replace (replace $id "id=\"" "") "\"" "" }}
                            {{- $header := replaceRE "<h[1-6].*?>((.|\n])+?)</h[1-6]>" "$1" $header -}}

                            {{- if ne $i 0 -}}
                            {{- $prevHeaderLevel := index (findRE "[1-6]" (index $headers (sub $i 1)) 1) 0 -}}
                            {{- $prevHeaderLevel := len (seq $prevHeaderLevel) -}}
                            {{- if gt $headerLevel $prevHeaderLevel -}}
                            {{- range seq $prevHeaderLevel (sub $headerLevel 1) -}}
                            <ul>
                                {{/* the first should not be recorded */}}
                                {{- if ne $prevHeaderLevel . -}}
                                {{- $.Scratch.Add "bareul" . -}}
                                {{- end -}}
                                {{- end -}}
                                {{- else -}}
                                </li>
                                {{- if lt $headerLevel $prevHeaderLevel -}}
                                {{- range seq (sub $prevHeaderLevel 1) -1 $headerLevel -}}
                                {{- if in ($.Scratch.Get "bareul") . -}}
                            </ul>
                            {{/* manually do pop item */}}
                            {{- $tmp := $.Scratch.Get "bareul" -}}
                            {{- $.Scratch.Delete "bareul" -}}
                            {{- $.Scratch.Set "bareul" slice}}
                            {{- range seq (sub (len $tmp) 1) -}}
                            {{- $.Scratch.Add "bareul" (index $tmp (sub . 1)) -}}
                            {{- end -}}
                            {{- else -}}
                        </ul>
                        </li>
                        {{- end -}}
                        {{- end -}}
                        {{- end -}}
                        {{- end -}}
                        {{- end }}
                        <li>
                            <a href="#{{- $cleanedID -}}" aria-label="{{- $header | plainify -}}">{{- $header | safeHTML -}}</a>
                            {{- else }}
                        <li>
                            <a href="#{{- $cleanedID -}}" aria-label="{{- $header | plainify -}}">{{- $header | safeHTML -}}</a>
                            {{- end -}}
                            {{- end -}}
                            {{- end -}}
                            <!-- {{- $firstHeaderLevel := len (seq (index (findRE "[1-6]" (index $headers 0) 1) 0)) -}} -->
                            {{- $firstHeaderLevel := $largest }}
                            {{- $lastHeaderLevel := len (seq (index (findRE "[1-6]" (index $headers (sub (len $headers) 1)) 1) 0)) }}
                        </li>
                        {{- range seq (sub $lastHeaderLevel $firstHeaderLevel) -}}
                        {{- if in ($.Scratch.Get "bareul") (add . $firstHeaderLevel) }}
                    </ul>
                    {{- else }}
                    </ul>
                    </li>
                    {{- end -}}
                    {{- end }}
                    </ul>
                </div>
            </details>
        </div>
    </aside>
    <script>
        let activeElement;
        let elements;
        window.addEventListener('DOMContentLoaded', function (event) {
            checkTocPosition();

            elements = document.querySelectorAll('h1[id],h2[id],h3[id],h4[id],h5[id],h6[id]');
            // Make the first header active
            activeElement = elements[0];
            const id = encodeURI(activeElement.getAttribute('id')).toLowerCase();
            document.querySelector(`.inner ul li a[href="#${id}"]`).classList.add('active');
        }, false);

        window.addEventListener('resize', function(event) {
            checkTocPosition();
        }, false);

        window.addEventListener('scroll', () => {
            // Check if there is an object in the top half of the screen or keep the last item active
            activeElement = Array.from(elements).find((element) => {
                if ((getOffsetTop(element) - window.pageYOffset) > 0 && 
                    (getOffsetTop(element) - window.pageYOffset) < window.innerHeight/2) {
                    return element;
                }
            }) || activeElement

            elements.forEach(element => {
                const id = encodeURI(element.getAttribute('id')).toLowerCase();
                if (element === activeElement){
                    document.querySelector(`.inner ul li a[href="#${id}"]`).classList.add('active');
                } else {
                    document.querySelector(`.inner ul li a[href="#${id}"]`).classList.remove('active');
                }
            })
        }, false);

        const main = parseInt(getComputedStyle(document.body).getPropertyValue('--article-width'), 10);
        const toc = parseInt(getComputedStyle(document.body).getPropertyValue('--toc-width'), 10);
        const gap = parseInt(getComputedStyle(document.body).getPropertyValue('--gap'), 10);

        function checkTocPosition() {
            const width = document.body.scrollWidth;

            if (width - main - (toc * 2) - (gap * 4) > 0) {
                document.getElementById("toc-container").classList.add("wide");
            } else {
                document.getElementById("toc-container").classList.remove("wide");
            }
        }

        function getOffsetTop(element) {
            if (!element.getClientRects().length) {
                return 0;
            }
            let rect = element.getBoundingClientRect();
            let win = element.ownerDocument.defaultView;
            return rect.top + win.pageYOffset;   
        }
    </script>
    {{- end }}
    ```

- Update CSS:

    ```css
    /* assets/css/extended/blank.css */
    :root {
        --nav-width: 1380px;
        --article-width: 650px;
        --toc-width: 300px;
    }
    .toc {
        margin: 0 2px 40px 2px;
        border: 1px solid var(--border);
        background: var(--entry);
        border-radius: var(--radius);
        padding: 0.4em;
    }
    .toc-container.wide {
        position: absolute;
        height: 100%;
        border-right: 1px solid var(--border);
        left: calc((var(--toc-width) + var(--gap)) * -1);
        top: calc(var(--gap) * 2);
        width: var(--toc-width);
    }
    .wide .toc {
        position: sticky;
        top: var(--gap);
        border: unset;
        background: unset;
        border-radius: unset;
        width: 100%;
        margin: 0 2px 40px 2px;
    }
    .toc details summary {
        cursor: zoom-in;
        margin-inline-start: 20px;
        padding: 12px 0;
    }
    .toc details[open] summary {
        font-weight: 500;
    }
    .toc-container.wide .toc .inner {
        margin: 0;
    }
    .active {
        font-size: 110%;
        font-weight: 600;
    }
    .toc ul {
        list-style-type: circle;
    }
    .toc .inner {
        margin: 0 0 0 20px;
        padding: 0px 15px 15px 20px;
        font-size: 16px;
    }
    .toc li ul {
        margin-inline-start: calc(var(--gap) * 0.5);
        list-style-type: none;
    }
    .toc li {
        list-style: none;
        font-size: 0.95rem;
        padding-bottom: 5px;
    }
    .toc li a:hover {
        color: var(--secondary);
    }
    ```

## Sort posts by `LastMod`

### Posts page

- Update `layouts/_default/list.html`:

    ```html {hl_lines=[7]}
    <!-- layouts/_default/list.html -->
    {{- if .IsHome }}
    {{- $pages = where site.RegularPages "Type" "in" site.Params.mainSections }}
    {{- $pages = where $pages "Params.hiddenInHomeList" "!=" "true"  }}
    {{- end }}

    {{- $pages := $pages.ByLastmod.Reverse }} <!-- add this line -->

    {{- $paginator := .Paginate $pages }}
    ```

### Archives page

- Copy `themes/PaperMod/layouts/_default/archives.html` to `layouts/_default`, and change:

    ```html
    <!-- layouts/_default/archives.html -->
    {{- range $pages.GroupByPublishDate "2006" }}
        ...
        {{- range .Pages.GroupByDate "January" }}
        ...
            {{- range .Pages }}
            ...
            {{- end }}
            ...
        {{- end }}
        ...
    {{- end }}
    ```

    to:

    ```html {hl_lines=[2, 4, 6]}
    <!-- layouts/_default/archives.html -->
    {{- range $pages.GroupByParamDate "lastmod" "2006" }}
        ...
        {{- range .Pages.GroupByParamDate "lastmod" "January" }}
        ...
            {{- range .Pages.ByLastmod.Reverse }}
            ...
            {{- end }}
            ...
        {{- end }}
        ...
    {{- end }}
    ```

---

<div class="zhihu-ref">
  <div class="zhihu-ref-title">References</div>
  <ol>
    <li><a href="https://www.lilmp.com/categories/小m平部落格整形手術/" target="_blank">小M平碎碎念-小M平部落格整形手術</a></li>
    <li><a href="https://blog.csdn.net/Xuyiming564445/article/details/122011603" target="_blank">CSDN-Hugo博客PaperMod主题目录放在侧边</a></li>
  </ol>
</div>
