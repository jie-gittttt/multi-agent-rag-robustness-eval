"""从中文维基百科 API 批量拉取文章，保存为 markdown（简体）"""
import requests
import urllib3
import re
import time
from pathlib import Path

urllib3.disable_warnings()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
API_URL = "https://zh.wikipedia.org/w/api.php"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 已成功下载的不重复处理
EXISTING = {f.stem for f in OUT_DIR.glob("*.md")}

# 全部关键词（第一批成功的会跳过）
KEYWORDS = [
    "太阳能", "风能", "可再生能源", "光伏发电", "碳中和",
    "碳排放交易", "温室效应", "全球变暖", "气候变化", "巴黎协定",
    "碳捕获与封存", "生物质能", "地热能", "储能技术", "锂离子电池",
    "电动汽车", "氢能", "化石燃料", "净零排放", "碳足迹",
    "碳税", "能源转型", "水力发电", "核能", "清洁发展机制",
    "京都议定书", "海洋能", "智能电网", "能源政策", "核聚变",
]


def clean_wiki_text(text: str) -> str:
    """清理 wiki 文本中的标记"""
    # 去掉 == 标题标记（保留文字）
    text = re.sub(r"==+\s*(.+?)\s*==+", r"## \1", text)
    # 去掉 <ref> 引用
    text = re.sub(r"\[\d+\]", "", text)
    return text


def fetch_article(title: str) -> tuple[str, str] | None:
    """获取单篇文章，返回 (title, content) 或 None"""
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "format": "json",
        "explaintext": 1,
        "redirects": 1,
        "variant": "zh-cn",
    }
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30, verify=False)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for pid, pdata in pages.items():
            if pid == "-1":
                return None  # 页面不存在
            text = pdata.get("extract", "")
            if len(text) < 200:
                return None  # 内容太短（可能是消歧义页或重定向）
            return pdata.get("title", title), text
    except Exception as e:
        print(f"  FAIL {title}: {e}")
        return None


ok, skip = 0, 0
for kw in KEYWORDS:
    result = fetch_article(kw)
    if result is None:
        print(f"  SKIP {kw}")
        skip += 1
        continue

    title, content = result
    content = clean_wiki_text(content)
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", title)
    fpath = OUT_DIR / f"{safe_name}.md"
    fpath.write_text(f"# {title}\n\n{content}", encoding="utf-8")
    print(f"  OK  {title} ({len(content)} chars)")
    ok += 1

total = len(list(OUT_DIR.glob("*.md")))
print(f"\nDone. OK={ok} SKIP={skip} Total files in {OUT_DIR}: {total}")
