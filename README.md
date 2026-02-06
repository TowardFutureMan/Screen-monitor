# Attention Monitor

一个基于摄像头的注意力监控工具，实时检测用户是否偏离屏幕、低头玩手机或出现手机等“非工作状态”。

## 功能特性
- ✅ 实时摄像头监控
- ✅ 头部姿态检测（Pitch/Yaw）
- ✅ 手部抬起检测（疑似拿手机/吃东西）
- ✅ 手机识别（Object Detection）
- ✅ 中文语音提醒：“别摸鱼了，别摸鱼了”
- ✅ 支持 macOS 打包成 App

专注状态：
<img width="1470" height="956" alt="0f03cc45a2e967b536fb4222a2e389d1" src="https://github.com/user-attachments/assets/a30cf314-a5b6-4a0e-ad94-2f316a3dc780" />
玩手机时物体检测：
<img width="1470" height="956" alt="f32c50c878bca2261ee5666a9addb5e3" src="https://github.com/user-attachments/assets/22aac7ef-f462-47eb-9647-06a4d3113820" />
检测异常手部抬高：
<img width="1470" height="956" alt="0eb7879ecb0af985f57cb6064611c8c5" src="https://github.com/user-attachments/assets/5ea31d67-0984-4cc2-b182-2040091da039" />

## 运行环境
- Python 3.11
- macOS (M 系列测试通过)

## 安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 本地运行教程（小白版）

## 1. 准备环境
请确保电脑已安装 **Python 3.11**（推荐）  
如果没有，请先安装：  
👉 https://www.python.org/downloads/

安装后打开终端，输入：
python3 --version

如果显示 Python 3.11.x 就可以继续。

## 2. 下载项目
打开终端，输入：
git clone https://github.com/TowardFutureMan/Screen-monitor.git
cd Screen-monitor

## 3. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

## 4. 安装依赖
pip install -r requirements.txt

## 5. 运行程序
python attention_monitor.py
运行后会弹出窗口，点击“开始监控”即可。

6. 摄像头权限
第一次运行会弹出摄像头权限提示，请点击 允许。
如果没有弹出：
打开 系统设置 → 隐私与安全性 → 摄像头
勾选 Terminal / Cursor / VS Code


## （可选）打包成 App
bash scripts/build_mac_app.sh
生成文件位置：
dist/AttentionMonitor.app
第一次打开 App，请右键 → 打开。
