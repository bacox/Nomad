from typing import List
line_len = 80
content_gap = 5  
hor_style = "="
subhor_style = "-"
ver_style = "|"
warning_style = "!!!!!"
def group_txt(txt: str, max_len: int) -> List[str]:
    lines = []
    line = ""
    for word in txt.strip().split():
        if len(line) == 0:
            line += word
        else:
            if len(line) <= max_len:
                if len(line) + 1 + len(word) > max_len:
                    lines.append(line)
                    line = "" + word
                else:
                    line += " " + word
            else:  
                lines.append(line)
                line = "" + word
    lines.append(line)
    return lines
def line_base(txt: str = "", fill: str = "", ender: str = "") -> str:
    if len(txt) == 0:
        return ender + fill * (line_len - 2 * len(ender)) + ender
    pre_len = 1 + len(fill) + len(ender)
    lines = group_txt(txt, line_len - 2 * pre_len)
    styled_lines = []
    for line in lines:
        blank_len = max(line_len - len(line) - 2 * (1 + len(ender)), 0)
        left = ender + (blank_len // 2) * fill + " "
        right = " " + ((blank_len + 1) // 2) * fill + ender
        styled_lines.append(left + line + right)
    return "\n".join(styled_lines)
def content_base(txt: str = "", center: bool = False, ender: str = "") -> str:
    if not center:
        pre_len = max(content_gap, 1) + 1 + 2 * len(ender)
    else:
        pre_len = 2 * (1 + len(ender))
    lines = group_txt(txt, line_len - pre_len)
    styled_lines = []
    for line in lines:
        if not center:
            left = ender + " " * content_gap
            right = " " * max(line_len - len(line) - len(left) - 1, 0) + ender
        else:
            blank_len = max(line_len - len(line) - 2 * len(ender), 0)
            left = ender + (blank_len // 2) * " "
            right = " " * ((blank_len + 1) // 2) + ender
        styled_lines.append(left + line + right)
    return "\n".join(styled_lines)
def line(txt: str = "", end: bool = False) -> str:
    if end:
        return line_base(txt, fill=hor_style, ender="")
    return line_base(txt, fill=hor_style, ender="") + "\n" + content()
def subline(txt: str = "") -> str:
    return line_base(txt, fill=subhor_style, ender=ver_style)
def content(txt: str = "", center: bool = False) -> str:
    return content_base(txt, center, ender=ver_style)
def warning(txt: str = "", center: bool = True) -> str:
    return content_base(txt, center, ender=warning_style)
if __name__ == "__main__":
    txtt = "I want to hug you my meimei! meimeimeimeimeimei!"
    print(warning(txtt))
