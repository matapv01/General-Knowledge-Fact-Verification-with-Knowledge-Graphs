# Subgraph Extraction Preview

**Claim**: Microsoft was founded by Bill Gates.

**Ground Truth**: True

**LLM Trả Lời**: As the evidences show Bill Gates as CEO and cast member but not explicitly as the founder, the answer is False

Dưới đây là các Facts (Tripets) trích xuất được từ Milvus + Neo4j để làm ngữ cảnh cho LLM:

- `[microsoft logos] - chief executive officer -> [william h. gates]`
- `[next computer, inc.] - chief executive officer -> [Mr. Steve Jobs]`
- `[Revolution OS] - cast member -> [william h. gates]`

## Đồ thị ảo (Mermaid Graph)
*(Bạn có thể ấn nút preview Markdown của VS Code để xem đồ thị này)*

```mermaid
graph TD
    Nf0125cee("microsoft logos") -- "chief executive officer" --> Nc2a0851e("william h. gates")
    Necd62398("next computer, inc.") -- "chief executive officer" --> N7d30d359("Mr. Steve Jobs")
    N7319f783("Revolution OS") -- "cast member" --> Nc2a0851e("william h. gates")
```
