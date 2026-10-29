# PC 端使用

PC 入口位于 `scripts/run_pc.py`，运行独立的文本对话页面。它读取现有 `PersonaCard`、`DialogOrchestrator` 和 provider 代码，使用 `pc/providers.yaml` 选择模型，不修改板端的 `providers.yaml`。它不启动摄像头、ASR、TTS 或板端 WebSocket 服务。页面的回复来自所选真实对话 provider；没有有效密钥时不会产生模拟回答。

## 快速启动

先在 PowerShell 进入仓库根目录，即包含 `README.md` 和 `scripts/` 的目录。若用浏览器下载 ZIP，先解压。随后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-pc.txt
$env:DASHSCOPE_API_KEY = Read-Host 'DashScope API Key'
.\.venv\Scripts\python.exe scripts/run_pc.py
```

打开 `http://127.0.0.1:8765/`。默认自动查找已设置密钥的平台，按 DashScope、DeepSeek、智谱、OpenAI 的顺序尝试；上面的示例只设置了 DashScope。关闭时在运行服务的终端按 `Ctrl+C`。端口已被使用时，追加 `--port 8766`，浏览器也改用对应端口。密钥只保存在当前 PowerShell 进程的环境变量中，不能写进 `providers.yaml` 或提交到 GitHub。调用云端服务可能产生费用。

页面提示设置 API Key 表示启动服务的终端没有检测到对应平台的密钥；发送消息后提示鉴权失败表示平台拒绝了当前密钥。`GET /api/status` 中的 `ready` 只检查密钥是否已设置，不向云端发送验证请求。连接或模型出错时，页面显示错误，不会把人格卡中的预设回复伪装成真实回答。

## 更换角色

复制公开示例，修改人格卡，使用 `--character` 指向新目录：

```powershell
Copy-Item characters/example characters/my-character -Recurse
.\.venv\Scripts\python.exe scripts/run_pc.py --character .\characters\my-character
```

编辑 `characters/my-character/persona.yaml` 中的 `name`、`profile`、`style` 和 `emotions`。`profile` 进入主对话提示，`style` 给本地口吻改写模块使用。默认 PC 快速模式不运行本地改写模型，中文回复显示云端原文；日语人格卡会自动使用本地日语敬体转换规则。日语角色需设置 `language: ja`、日语 `profile.speech_samples` 和合适的 `style.sentence_endings`。可按需填写 `examples.jsonl`；它用于中文本地改写。自定义角色目录默认由 `.gitignore` 排除。

角色立绘放在 `characters/my-character/sprites/aligned/`，文件名由 `persona.yaml` 的 `emotions` 指定。当前情绪键为 `neutral`、`happy`、`surprised`、`shy`、`unhappy`、`assertive`。PC 页面按映射查找同名 `.webp`，再查找 `.png`；没有可用立绘时显示角色首字占位，不报“立绘加载失败”。请只使用有权展示的形象素材。

## 选择对话服务

启动时不必指定模型，也不必指定平台。每个平台的模型和接口地址在 `pc/providers.yaml` 中配置；API 请求必须带模型名。只设置一个平台的密钥时，会自动使用该平台；设置多个密钥时，会按配置顺序尝试，直到有一个返回有效响应。页面初始状态列出待尝试的平台，回答后显示实际成功的平台。也可通过 `--provider` 只使用一个平台。密钥必须来自对应 API 平台；把 DeepSeek 密钥写入 `DASHSCOPE_API_KEY` 不会生效。

DeepSeek 使用 `pc/providers.yaml` 中的 `deepseek-flash`：

```powershell
$env:DEEPSEEK_API_KEY = Read-Host 'DeepSeek API Key'
.\.venv\Scripts\python.exe scripts/run_pc.py
```

OpenAI API 密钥使用以下命令。ChatGPT 的订阅账号不等于 OpenAI API 密钥或 API 额度：

```powershell
$env:OPENAI_API_KEY = Read-Host 'OpenAI API Key'
.\.venv\Scripts\python.exe scripts/run_pc.py --provider openai
```

智谱可用以下命令：

```powershell
$env:ZHIPU_API_KEY = Read-Host 'Zhipu API Key'
.\.venv\Scripts\python.exe scripts/run_pc.py --provider zhipu
```

`GET /api/status` 会显示当前候选平台和模型；出现 401 时先核对密钥来源是否对应这些平台。若多个已设置密钥的平台都失败，页面会显示错误，不会产生模拟回复。更改环境变量或人格卡后重启 PC 服务。页面“新对话”按钮会清空当前进程中的历史记录，服务重启也会清空。PC 快速入口只供单人本机使用；它将对话文本、人格设定和有限轮次上下文发送给选定的云端服务，不发送立绘或本地图片。

## 本地口吻改写

中文角色若需要运行自研口吻改写模块，先自行准备合法使用的 GGUF 模型和 llama.cpp 服务，使 `pc/providers.yaml` 的 `local.completion_url` 可访问，再加上 `--local-rewrite`：

```powershell
.\.venv\Scripts\python.exe scripts/run_pc.py --local-rewrite
```

本仓库不包含模型权重。未启动本地模型时不要加此参数；若本地改写调用失败，页面仍显示云端原文并标明改写不可用。日语角色无需该参数，敬体转换由本地规则脚本完成。

## 完整设备链路

PC 入口提供文本对话和可选立绘展示。语音输入、语音输出、摄像头感知、播放确认和板端资源调度仍由原有的 `scripts/run_service.py` 提供，需要分别准备模型、设备、角色素材与授权。PC 页面运行成功不等于这些硬件链路已在板端验证。
