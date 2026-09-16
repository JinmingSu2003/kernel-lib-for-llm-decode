# 实现路线与验收门槛

预计节奏仅用于排期：有 PyTorch 基础、能稳定使用 GPU 时约 6–8 周；不构成完成时间保证。按验收门槛推进。

| 阶段 | 工作 | 完成标准 |
|---|---|---|
| M0：参考与环境 | 选定 GPU、固定版本、PyTorch 显式 Attention/RMSNorm/SiLU 参考 | CPU 小规模数学测试通过；GPU 基线、环境记录可复现 |
| M1：基础算子 | RMSNorm、SiLU×Mul，之后 Add RMSNorm | 尾块、不同精度、残差输出验证通过；记录 eager 与适用 compile 基线 |
| M2：连续 Decode | 每请求每头起步、分块、Online Softmax | GQA 映射、变长 mask、所有目标头维度通过；具备延迟曲线 |
| M3：Split-KV | 分段统计量、稳定合并、显式配置 | 空分段和极端值测试通过；明确何时更快、何时更慢 |
| M4：分页 Decode | 页表寻址、随机页、页尾 mask | 与 gather 后的独立参考一致；计时路径不 gather KV |
| M5：缓存管理 | 预留、追加、回收、容量不足处理 | 跨页增长、释放再用、多请求多层测试通过 |
| M6：模型闭环 | 固定 checkpoint、Prefill 导入、逐步 Decode | 固定 token 输入的多步 logits 对齐；消融报告完成 |

## 最低完整交付

RMSNorm + SiLU×Mul + 连续与分页 GQA Decode + 页池生命周期 + 一个模型的多步对齐和性能报告。Split-KV 可在 M2 后开发；时间不足时先完成分页和模型闭环，将 Split-KV 留作增强。

## 每个优化的记录方式

1. 提出具体假设，例如“小 batch 长序列时 program 数不足”。
2. 保存可复现的优化前实现和配置。
3. 只改变可解释的一组变量，记录正确性和性能变化。
4. 用 profiler 和数据判断假设；失败结果也保留。
5. 在目标形状之外检查是否退化，再决定默认 dispatch。

## 完成后再考虑

同组 Query 头复用 KV、分页 Split-KV、shape dispatch、真实动态请求调度、上游贡献。量化、前缀共享和多 GPU 分别属于独立扩展，不与首版范围混在一起。
