# Attention Monitor

一个基于摄像头的注意力监控工具，实时检测用户是否偏离屏幕、低头玩手机或出现手机等“非工作状态”。

## 功能特性
- ✅ 实时摄像头监控
- ✅ 头部姿态检测（Pitch/Yaw）
- ✅ 手部抬起检测（疑似拿手机/吃东西）
- ✅ 手机识别（Object Detection）
- ✅ 中文语音提醒：“别摸鱼了，别摸鱼了”
- ✅ 支持 macOS 打包成 App

## 运行环境
- Python 3.11
- macOS (M 系列测试通过)

## 安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
