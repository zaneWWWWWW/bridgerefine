# GitHub 发布记录

## 补充上传：完整材料

用户随后要求将此前未上传的手稿等材料全部上传。新材料入口为仓库 `materials/`，完整数据与大文件通过同一私有仓库的 Release 分包提供：

- 材料 Release：https://github.com/zaneWWWWWW/bridgerefine/releases/tag/materials-2026-09-17
- 范围：整理版中的手稿、图表、全部数据、权重、预测、说明、历史稿和整理记录；原工作区冗余的旧 ZIP 不重复上传。
- 普通 Git：可浏览手稿、常用图、结果 CSV/JSON 和说明。
- Release：全部非代码材料的独立 ZIP 分包、清单及 SHA-256。代码继续保存在仓库根目录。
- 访问权限保持私有。
- 下载恢复：在克隆仓库中运行 `python3 tools/download_materials.py`。

以下为最初仅上传代码时的历史记录，不代表补充上传后的材料范围。

## 首次上传：代码快照

- 仓库：https://github.com/zaneWWWWWW/bridgerefine
- 可见性：**PRIVATE（私有）**
- 默认分支：`main`
- 提交：`2799bce744fa62da23945cb7d05cef05f96f132b`
- 提交说明：`Archive BridgeRefine research code and add result verification tools`
- 本地目录：`02_代码/bridgerefine/`
- 发布日期：2026-09-17
- 已推送：80 个代码/配置/说明文件，提交内容共 431,767 字节。

远端 `main` 与本地提交一致，工作区干净。已从 GitHub 重新克隆，71 份原始代码/配置来源哈希校验通过；克隆版本的汇总工具在同一批本地结果上生成了完全相同的输出。详见 `GitHub克隆验证.json`。

未上传医学影像、患者划分文件、逐患者指标、模型权重、手稿 PDF/Word 或图像。原始代码中的算法和路径保持归档原貌。仓库记录的是本次整理快照，不是既有模型的历史训练提交。

克隆命令：

```bash
gh repo clone zaneWWWWWW/bridgerefine
```

后续完整复现所需补件已列在仓库 README 和本地 `版本核对与缺项.md` 中。
