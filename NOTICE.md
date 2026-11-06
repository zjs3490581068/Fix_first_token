# 来源与移植说明

本项目提供固定首 Token 数学推理实验的搜索、生成、判题与统计实现。

参考项目：

- `sophicle/cues`，提交 `cd9caabe748307df0565502f630e3a834a9f998d`，MIT 许可原文保存在 `LICENSE.cues`。
- `limit-of-RLVR`，提交 `79c348f4543330bb78b01a5332df09fea2700f70`，用于核对多次采样协议与 `math/pass@k.py` 的估计公式。

移植与适配：

- `core.py` 中的 Boxed 提示词与停止规则来自 cues 的提示词配置。
- `search.py` 与 `grading.py` 的候选提出、答案分组及无标签熵筛选参考 cues 的 beam、search 与 agree 实现。
- `grading_base.py` 的完整括号答案提取、`metrics.py` 的 pass@k 与 bootstrap 函数适配自 cues 的 grader 与 metrics。
- `sampling.py` 参考 cues.evaluate，使用 Token ID 预填与 vLLM 采样；多卡协调、恢复、盲评样例抽取与补充实验入口为本地实现。

相关论文：Wang 等《Base Models Can Reason By Taking a Cue From Training Data》；Yue 等《Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?》；Chen 等《Evaluating Large Language Models Trained on Code》。
