# BatteryGuard

[English](README.md) · [项目主页](https://dingyucanada.github.io/BatteryGuard/) · [项目规格](docs/PROJECT_SPEC.md) · [三分钟演示脚本](docs/DEMO_RUNBOOK.md)

> **仅限仿真 / 研究。** BatteryGuard 是离线研究原型，不得连接真实充电器、电池测试仪、BMS、车辆、电池架或储能系统。

BatteryGuard 把前 30 个有效循环转化为一条可审计的研发证据链：寿命点估计与 split-conformal 区间、OOD 分数与拒答决定、非因果风险画像、充电策略仿真、确定性安全否决、一次性受控 synthetic 揭示，以及追加式哈希链证据记录。

## 为什么不只输出一个寿命数字

单点预测无法说明数据是否越界、区间是否过宽、输入是否缺失，也无法阻止优化器绕过安全边界。BatteryGuard 将以下结果设为同等重要的一等输出：

- 数据质量与未来信息泄漏硬门禁；
- B0 中位数、B1 线性模型、B2 XGBoost 的透明模型阶梯；
- 90% 名义目标的 split-conformal 区间；
- Mahalanobis OOD 与显式 ABSTAIN；
- 可观察风险画像与最近参考电芯；
- FAST、BALANCED、LIFE 与保守 fallback 的离线仿真；
- 唯一有权返回 `ALLOW` 的确定性 `SafetyShield`；
- 鉴权后一次性 synthetic 揭示与可验证追加式 Evidence Ledger。生成器与结果均为公开源码，因此这是访问控制演练，不是独立秘密盲测。

```text
早期循环 → 质量/泄漏门禁 → 模型阶梯 → conformal 区间
         → OOD/拒答 → 风险画像 → 策略候选 → Twin-0 仿真
         → SafetyShield → Pareto 集 → 受控 synthetic 揭示 → Evidence Ledger
```

## 五分钟启动

要求：Python 3.11–3.13 与 [`uv`](https://docs.astral.sh/uv/)。

```bash
uv sync --extra dev
uv run pytest
uv run batteryguard demo --cell random --seed 42 --offline --no-reveal
uv run streamlit run apps/streamlit_app.py
uv run uvicorn batteryguard.api.app:app --host 127.0.0.1 --port 8000
```

容器方式：

```bash
docker build -f docker/Dockerfile -t batteryguard:local .
export BATTERYGUARD_REVEAL_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose -f docker/compose.yaml up --build
```

CLI 默认不揭示。只有显式生成高熵 token 并传入 `--reveal`，才会运行 synthetic 访问控制路径：

```bash
export BATTERYGUARD_REVEAL_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run batteryguard demo --cell random --seed 42 --offline --reveal
```

未设置环境变量时，每个进程使用不对外公开的临时随机 token，外部揭示请求默认关闭；Compose 强制要求显式 token，且端口只绑定 `127.0.0.1`。这些是本地防护，不是生产级身份认证，不得把研究 API 暴露给不可信网络。

## 当前工程证据

0.1.0 release candidate 的带日期验证快照覆盖单元、集成、回归、隐私、发布加固与安全测试，并通过 Ruff、严格 Mypy 与非 root Linux ARM64 容器的 CLI/API/UI smoke test。CI 已配置 Python 3.11–3.13 矩阵和标准 Dockerfile 冷构建门禁；确切测试数、覆盖率与本机 Docker 网络限制见 [验证报告](docs/TEST_REPORT.md)。

合成 locked test（仅 8 个合成测试 cell）上，B0/B1/B2 的 MAE 分别为 180.625、63.200、33.691 cycles。名义 90% split-conformal 区间在这 8 个合成 cell 上观测到 100% coverage，平均宽度 298.505 cycles。这些数字只证明离线软件夹具可复现，不能外推为真实电芯、MATR、跨协议或跨化学体系表现。

完整命令、样本量、边界与故障注入证据见 [验证报告](docs/TEST_REPORT.md)。

MATR adapter 面向 Severson 等人的论文 [“Data-driven prediction of battery cycle life before capacity degradation”](https://doi.org/10.1038/s41560-019-0356-8) 所关联的数据结构。仓库不包含 MATR 结果或数据；使用者必须从原始分发获取数据，并核验其当前许可与引用要求。

HTTP ingest 只接受配置好的 `data/raw` 根目录下、经过 symlink 解析后仍位于其中的相对目录；绝对路径、`..` 逃逸和 symlink 逃逸都会被拒绝。其他可信本地路径只能走显式 CLI 导入流程。

## 不可绕过的边界

- 同一个 `cell_id` 不得跨 split；校准集不训练点模型，external OOD 不参与选择。
- `cycle_life`、未来周期、原始协议标识和其他寿命代理不得进入特征。
- 每个预测都带区间、OOD、拒答与证据 ID。
- 学习模型和优化器均无权写 `ALLOW`；最终权限属于 `SafetyShield`。
- 运行时公共 cell 载荷不含 `cycle_life`，揭示端点要求有效 token；由于 synthetic 生成器公开，这只验证软件访问控制，不证明标签保密。
- 仿真失败、NaN、缺失轨迹或硬约束越界进入 `REJECT`/`FALLBACK`，不会被包装成成功。

## 项目结构

- `src/batteryguard/ingestion`、`quality`、`features`：数据合同与泄漏门禁
- `prediction`、`uncertainty`、`ood`、`diagnosis`：预测、校准与风险证据
- `simulator`、`optimizer`、`safety`：策略仿真、Pareto 与最终否决
- `evidence`、`demo`：追加式账本与受控 synthetic 揭示
- `api`、`apps/streamlit_app.py`、`cli.py`：API、UI 与命令行入口
- `tests/`：单元、集成、回归与安全故障注入
- `site/`：GitHub Pages 静态项目主页

## 已知限制

- 仓库内数据全部为确定性合成夹具，不构成电池科学验证。
- 静态站点和源码公开了 synthetic 演示结果；前端遮罩按钮不是安全控件，也不构成独立盲测。
- Twin-0 指标是透明工程代理，不是特定商业电芯的经认证电化学预测。
- 风险画像只描述可观察关联，不证明 SEI、析锂、LAM 或 LLI。
- 本地账本可发现篡改，但没有外部信任锚；高保证场景需要签名、远程见证或 WORM 存储。
- 项目没有硬件接口。任何物理实验前仍需电芯专属标定、独立验证、HIL、功能/预期功能安全、网络安全与适用法规评审。
- FastAPI 管理与 evidence 路由只适用于受信任本机，不具备生产级用户认证或多租户隔离。
- 发布制品必须从干净 checkout 构建（例如运行 `uv build`），不得直接压缩开发工作目录；ignore 规则不能替代对最终归档和完整 Git 历史的密钥、私有数据、模型、ledger 与本机路径扫描。

## 参与贡献

提交代码前请阅读 [贡献指南](CONTRIBUTING.md)、[安全政策](SECURITY.md) 与 [行为准则](CODE_OF_CONDUCT.md)。学术或技术引用信息见 [CITATION.cff](CITATION.cff)。

Apache-2.0 licensed. Copyright 2026 Yu Ding.
