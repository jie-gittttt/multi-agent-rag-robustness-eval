from datetime import datetime
from typing import List

from jinja2 import Template


SYSTEM_TEMPLATE = Template("""\
你是一个水泥生命周期评估（LCA）领域的专家助手。你的知识涵盖了：
- 水泥生产工艺（生料制备、熟料煅烧、水泥粉磨）
- 生命周期评估方法论（ISO 14040/14044）
- 碳排放计算与核算（IPCC 指南、中国水泥行业核算指南）
- 替代燃料与原材料的碳减排潜力
- 碳捕集利用与封存（CCUS）技术

回答规则：
1. 仅根据提供的上下文片段回答问题。如果上下文信息不足，请明确说明。
2. 在回答中使用 [1]、[2] 等标记引用来源。
3. 回答应专业、准确，使用中文。
4. 如果问题涉及数值或数据，必须引用具体来源。

当前日期：{{ current_date }}""")

USER_TEMPLATE = Template("""\
## 参考上下文

{% for chunk in chunks %}
[{{ loop.index }}] 来源: {{ chunk.metadata.filename }}
{{ chunk.content }}

{% endfor %}

## 对话历史

{% for msg in history %}
{{ msg.role }}: {{ msg.content }}
{% endfor %}

## 当前问题

{{ question }}

请基于以上上下文回答问题，并在适当位置标注引用编号。""")


class PromptBuilder:
    def build(self, question: str, chunks: list,
              history: List[dict] = None) -> list:
        system_content = SYSTEM_TEMPLATE.render(
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        user_content = USER_TEMPLATE.render(
            chunks=chunks,
            history=history or [],
            question=question,
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
