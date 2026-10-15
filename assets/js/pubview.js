/* 出版物列表的「卡片 / IEEE 列表」视图切换。
 *
 * 状态只存在 <html data-pubview> 这个属性上：CSS 靠它决定显示哪个视图，
 * 没有属性时就是默认的卡片视图，所以禁用 JS 也不会只剩空白。
 * 选择记进 localStorage，下次进页面按上次的视图渲染。
 */
(function () {
    var root = document.documentElement;
    var KEY = "pubView";
    var VIEWS = ["cards", "ieee"];

    function stored() {
        try {
            var v = localStorage.getItem(KEY);
            return VIEWS.indexOf(v) === -1 ? null : v;
        } catch (e) {
            // 隐私模式下 localStorage 可能直接抛异常，退回默认视图
            return null;
        }
    }

    var saved = stored();
    if (saved) root.setAttribute("data-pubview", saved);

    document.addEventListener("DOMContentLoaded", function () {
        var buttons = document.querySelectorAll(".pub-viewswitch-btn");
        if (!buttons.length) return;

        // 切换按钮在 HTML 里是 hidden 的，脚本跑起来才显示，避免无 JS 时留下点不动的按钮
        var toolbar = document.querySelector(".pub-toolbar");
        if (toolbar) toolbar.hidden = false;

        function apply(view) {
            root.setAttribute("data-pubview", view);
            Array.prototype.forEach.call(buttons, function (btn) {
                var on = btn.getAttribute("data-pubview") === view;
                btn.classList.toggle("is-active", on);
                btn.setAttribute("aria-pressed", on ? "true" : "false");
            });
        }

        Array.prototype.forEach.call(buttons, function (btn) {
            btn.addEventListener("click", function () {
                var view = btn.getAttribute("data-pubview");
                apply(view);
                try {
                    localStorage.setItem(KEY, view);
                } catch (e) {
                    // 存不下就算了，本次点击已经生效
                }
            });
        });

        apply(root.getAttribute("data-pubview") || "cards");
    });
})();
