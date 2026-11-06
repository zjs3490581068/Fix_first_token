# Fix First Token

研究固定首个输出 Token 对大语言模型数学推理的影响。包含 Qwen3-4B-Base 的 Cue 搜索、MATH-500 推理采样、GRPO 对比、答案判题、pass@k 分析与盲评样例抽取。

完整配置、协议及命令说明见 [README](README.md)。数据、模型权重、生成结果和私有评分映射由使用者自行准备。

## 安装

推理需要 Linux、Python 3.11 和可运行对应 CUDA 版本 PyTorch/vLLM 的 NVIDIA GPU。

```bash
git clone https://github.com/zjs3490581068/Fix_first_token.git
cd Fix_first_token
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

仅运行 CPU 判题、分析与测试，可安装 `requirements-dev.txt`。

## 准备输入

在 `configs/main.json` 中指定两份模型、测试集和训练探针集的路径，以及 GPU 编号、显存和磁盘预算。示例采用仓库内的 `models/`、`data/`、`outputs/`，默认使用 GPU 0；这些数据目录不会提交到 Git。

也可复制为 `configs/local.json`，保存自己的配置。相对路径以声明它的配置文件所在目录为基准。英文首词与双 Token 配置使用 `extends` 继承公共参数。

测试集需要 500 道题，探针集至少 100 道题。输入为 JSONL，包含 `problem_key`、`problem`、`gt`、`level`、`capability`；格式示例见英文 README。题面之外的答案标签不进入生成或搜索接口。

## 运行

```bash
# 主实验：搜索、采样、判题、汇总
python -m first_token run --config configs/main.json --run-dir outputs/main

# 四种英文首词，每题一次
python -m experiments.english_tokens --config configs/english_tokens.json --run-dir outputs/english_tokens

# To determine 双 Token，每题一次
python -m experiments.two_token --config configs/two_token.json --run-dir outputs/two_token
```

主实验包含自由生成、中文“解答”、固定 ` To`、搜索赢家及 GRPO 五组。搜索赢家与 ` To` 相同时共享 C/D 样本。两个补充实验可用 `--baseline-run outputs/main` 读取主实验第 0 次采样作配对比较。

统一使用温度 0.6、top-p 0.95、16,384 Token 总输出预算，固定 ID 计入总预算。提示词、停止规则、搜索筛选和指标计算的细节见英文 README。

## 盲评样例与后台运行

```bash
python -m experiments.extract_judge_samples \
  --plan sampling_plan.md --run-dir outputs/main \
  --references data/math500_official.jsonl --out outputs/judge_samples

bash scripts/launch_tmux.sh main_run configs/main.json
```

抽样方案需提供固定 50 题的题号、难度和采样索引表。抽取器保留 A/B/C 共 150 条完整回答，将盲评输入与私有分组映射分开保存。参考解答文件及选择表均由使用者提供，抽取器不调用 AI 评分服务。

后台启动器使用当前已激活环境的 Python，要求系统已安装 tmux。运行清单、分片校验、日志和退出码支持恢复；改变配置或代码时应使用新运行目录。

## 测试与许可

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
# 包含真实 tokenizer 编码测试
python -m pytest -q --tokenizer-path=models/Qwen3-4B-Base
```

代码采用 [MIT 许可](LICENSE)，参考实现和原始版权声明见 [NOTICE.md](NOTICE.md) 与 [LICENSE.cues](LICENSE.cues)。
