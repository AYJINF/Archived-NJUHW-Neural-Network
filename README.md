# 基于CR-GAN、RaR与LAT的侧脸年龄编辑

- 具体作业内容与实现方式见论文`paper.pdf`（写得很认真，求老师助教手下留情qwq）
- 由于实现方式差异，作业中要求的`train.sh`为`./my_CR-GAN/train.py`，`test.sh`为`./my_CR-GAN/test.py`
- 除了辅助模型以外的`python files`均位于`./my_CR-GAN`
- 模型较大所以不上传了
- 本作业参考的辅助模型：
  - CR-GAN: https://github.com/bluer555/CR-GAN
  - Rotate-and-Render: https://github.com/Hangz-nju-cuhk/Rotate-and-Render?tab=readme-ov-file
  - Lifespan_Age_Transformation_Synthesis: https://github.com/royorel/Lifespan_Age_Transformation_Synthesis
- 本作业自行实现的网络结构：`./my_CR-GAN/model.py`
- LAT_CR-GAN的拼接脚本：`LAT_CR-GAN_demo.py`
- 部分结果展示：`./result`
