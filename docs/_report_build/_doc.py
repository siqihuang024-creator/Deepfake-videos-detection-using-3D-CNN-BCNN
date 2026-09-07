DOC = '''<title>3D-CNN 贝叶斯检测实验</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Noto+Serif+SC:wght@400;500;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
''' + STYLE + '''

<div class="page">

<header>
  <h1>3D-CNN 与贝叶斯单类头的深度伪造视频检测实验</h1>
  <p class="sub">实验方法、每一步的目的，以及实测数据</p>
  <p class="meta">实验期间 2026-09-04 — 2026-09-07 · 45 次运行 · 2180 个 epoch</p>
</header>

<div class="abstract">
  <div class="lead">本文档的范围</div>
  <p>本文档记录两组实验的<b>方法、步骤目的与实测数据</b>：网络结构、划分协议、预处理、训练配置，以及每一步得到的曲线、表格与指标。文档不含对这些数据的解释或结论。所有数值均由 <span class="m">results/</span> 下的文件计算，未经转录。</p>
</div>

<h2><span class="n">1</span>模型架构</h2>

<p>特征提取器沿用 Leyva 等人 (2024) 的 fine-to-coarse 结构并沿时间轴膨胀为 3D，后接两种形状相同的分类头，分别用于两个训练阶段。</p>

''' + ARCH + '''

<div class="tw">
<table>
  <caption><b>表 1.</b> 逐层输出尺寸与参数量，以人脸裁剪输入 3×8×256×256 为例。3D 卷积的时间维填充为 1，故时间长度全程保持 8；时间感受野为 7 帧。</caption>
  <thead><tr><th>层</th><th>算子</th><th class="n">输出尺寸</th><th class="n">参数量</th></tr></thead>
  <tbody>
    <tr><td>输入</td><td>片段，RGB</td><td class="n">3×8×256×256</td><td class="n">—</td></tr>
    <tr><td>卷积段 1</td><td>Conv3d 3→16, k=3×5×5</td><td class="n">16×8×252×252</td><td class="n">3,616</td></tr>
    <tr><td></td><td>AvgPool3d 1×4×4, 步长 1×2×2</td><td class="n">16×8×125×125</td><td class="n">—</td></tr>
    <tr><td></td><td>BatchNorm3d → ReLU</td><td class="n">16×8×125×125</td><td class="n">32</td></tr>
    <tr><td>卷积段 2</td><td>Conv3d 16→24, k=3×5×5</td><td class="n">24×8×121×121</td><td class="n">28,824</td></tr>
    <tr><td></td><td>AvgPool3d → BatchNorm3d → ReLU</td><td class="n">24×8×59×59</td><td class="n">48</td></tr>
    <tr><td>卷积段 3</td><td>Conv3d 24→32, k=3×5×5</td><td class="n">32×8×55×55</td><td class="n">57,632</td></tr>
    <tr><td></td><td>AvgPool3d → BatchNorm3d → ReLU</td><td class="n">32×8×26×26</td><td class="n">64</td></tr>
    <tr><td>时间池化</td><td>沿时间轴取均值（8 个时间位置合为 1）</td><td class="n">32×26×26</td><td class="n">—</td></tr>
    <tr><td>空间池化</td><td>AdaptiveAvgPool2d(22)</td><td class="n">32×22×22</td><td class="n">—</td></tr>
    <tr><td>输出归一化</td><td>BatchNorm2d(32) → 展平</td><td class="n">15488</td><td class="n">64</td></tr>
    <tr class="head"><td>提取器合计</td><td></td><td class="n">15488</td><td class="n">90,280</td></tr>
    <tr><td>阶段 A 头</td><td>Linear 15488→512 → Dropout → Linear 512→1</td><td class="n">1</td><td class="n">7,930,881</td></tr>
    <tr><td>阶段 B 头</td><td>同上形状，均值场贝叶斯</td><td class="n">1</td><td class="n">7,930,881 × 2</td></tr>
  </tbody>
</table>
</div>

<h2><span class="n">2</span>实验方法</h2>

<h3>2.1 划分协议</h3>

<p><b>目的：</b>使测试集中的身份从未在训练中出现，且伪造视频的源真实视频不在训练集内。</p>

<p>划分按身份不相交执行，并附加<b>供体约束</b>：一条伪造视频只有在其目标身份与供体身份<em>同时</em>落入该子集时才被纳入。DFD 的伪造视频命名遵循 <span class="m">&lt;target&gt;_&lt;source&gt;__&lt;scene&gt;__&lt;id&gt;</span>，两个身份均可从文件名恢复。若仅按视频划分，DFD 测试集中 73.2% 的伪造视频可在训练集中找到自己的源真实视频。该约束在 DFD 剔除 1103 条伪造视频（占 36%），在 CelebDF++ 剔除 14538 条（占 27.3%）。</p>

<h3>2.2 预处理</h3>

<p><b>目的：</b>把变长视频转成定长片段；两种输入模式作为实验设计中的自变量。</p>

<p><b>整帧模式（实验一）：</b>保持原生分辨率，沿两个空间轴各隔一个像素取一个（<span class="m">frame[::2, ::2]</span>），1920×1080 变为 960×540，长宽比不变，不含插值。</p>

<p><b>人脸裁剪模式（实验三）：</b>dlib HOG 检测器定位人脸，回归 81 个关键点，绕检测框中心旋转使两眼连线水平，按检测框外扩至 ×2.0 裁剪，缩放到 256×256，越界部分以边缘像素复制填充。检测在训练前离线执行一次并缓存为逐视频的坐标文件（<span class="m">frame_indices</span>、<span class="m">boxes</span>、<span class="m">landmarks</span>、<span class="m">detect_stride</span>），训练与评估只读取坐标。检测步长为 4 帧，中间帧由线性插值得到。</p>

<p>两种模式的片段长度均为 8 帧。</p>

<h3>2.3 两阶段训练</h3>

<p><b>阶段 A 的目的：</b>用真伪两类标签，以确定性梯度把特征提取器训练到能区分真伪；训练结束后丢弃分类头，只保留提取器权重。</p>

<p><b>阶段 B 的目的：</b>冻结提取器，只在真实视频上训练贝叶斯头，使其对真实视频给出高后验、对偏离该分布的输入给出低后验，异常分数取后验预测的负值。</p>

<div class="tw">
<table>
  <caption><b>表 2.</b> 两个阶段的训练配置。三个实验共用同一套设置，差异仅在数据集、输入模式与早停耐心（见表 3）。</caption>
  <thead><tr><th>项</th><th>阶段 A</th><th>阶段 B</th></tr></thead>
  <tbody>
    <tr><td>目标函数</td><td>二元交叉熵</td><td>TraceGraph ELBO（Pyro SVI）</td></tr>
    <tr><td>优化器</td><td>Adam</td><td>SGD</td></tr>
    <tr><td>学习率</td><td>1×10⁻⁴，指数衰减 γ = 0.95</td><td>1×10⁻⁴，指数衰减 γ = 0.95</td></tr>
    <tr><td>训练样本</td><td>真伪两类，每类每 epoch 采样 1000 个片段</td><td>仅真实视频</td></tr>
    <tr><td>batch</td><td>4</td><td>4</td></tr>
    <tr><td>epoch 上限</td><td>60</td><td>50</td></tr>
    <tr><td>提取器</td><td>可训练</td><td>冻结，含 BatchNorm 统计量</td></tr>
    <tr><td>checkpoint 选择</td><td>验证 AUROC 最高的一轮</td><td>验证 macro AUROC 最高的一轮</td></tr>
  </tbody>
</table>
</div>

<h3>2.4 评估指标</h3>

<p><b>目的：</b>在封存的测试子集上给出可与文献对照的视频级读数。</p>

<p>每条视频取 8 个等距片段，分数取其均值，因此所有指标均为 <b>video-level</b>。主指标为 <b>AUROC</b>，次要指标为 AP、Accuracy、Balanced Accuracy、EER 与 TPR@5%FPR。判定阈值在验证集的真实视频上按 5% 假阳率校准，随 checkpoint 一同保存。</p>

<p>AUROC 的置信区间由<b>按身份聚类的自助重采样</b>给出（2000 次抽样，重采样身份而非视频）：CelebDF++ 上一条真实片段平均衍生 84 条伪造视频，共享背景、服装、光照与镜头运动，按视频重采样会把它们当作独立观测。</p>

<h3>2.5 实验设计</h3>

<div class="tw">
<table>
  <caption><b>表 3.</b> 本文档记录的两组实验。网络结构、输入尺寸、片段长度、优化器与学习率调度两者一致。</caption>
  <thead><tr><th>实验</th><th>数据集</th><th>输入模式</th><th class="n">阶段A 早停耐心</th><th>状态</th></tr></thead>
  <tbody>
    <tr><td>实验一</td><td>DFD</td><td>整帧，隔点抽样</td><td class="n">12</td><td>完成</td></tr>
    <tr><td>实验三</td><td>CelebDF++</td><td>人脸裁剪 ×2.0</td><td class="n">8</td><td>完成</td></tr>
  </tbody>
</table>
</div>

<h2><span class="n">3</span>实验一：DFD 整帧输入</h2>

<p><b>目的：</b>在不做人脸检测的条件下，测量该两阶段检测器在 DFD 上的端到端性能。</p>

<div class="tw">
<table>
  <caption><b>表 4.</b> DFD 划分统计。标签约定：1 = 真实，0 = 伪造。DFD 含 363 条原始视频与 3068 条伪造视频，由 28 名演员录制。</caption>
  <thead><tr><th>子集</th><th class="n">真实</th><th class="n">伪造</th><th class="n">身份数</th><th>用途</th></tr></thead>
  <tbody>
    <tr><td>训练</td><td class="n">264</td><td class="n">1894</td><td class="n">20</td><td>阶段 A / 阶段 B</td></tr>
    <tr><td>验证</td><td class="n">45</td><td class="n">17</td><td class="n">4</td><td>checkpoint 选择</td></tr>
    <tr><td>测试</td><td class="n">54</td><td class="n">54</td><td class="n">4</td><td>最终评估，全程封存</td></tr>
    <tr><td class="dim">供体约束剔除</td><td class="n dim">0</td><td class="n dim">1103</td><td class="n dim">—</td><td class="dim">不参与任何阶段</td></tr>
  </tbody>
</table>
</div>

''' + fig(EXP1_LOSS, 2,
          '实验一阶段 A 的训练交叉熵，60 个 epoch，自 1.5778 降至 0.6579。虚线为 <b>ln 2 = 0.6931</b>，即在类别平衡的数据上对所有输入恒定输出 0.5 的平凡预测器所得的交叉熵；阶段 A 每类每 epoch 各采样 1000 个片段，两类平衡，故该参照适用。60 个 epoch 中有 46 个位于该线之下。鼠标悬停可读取任一 epoch 的数值。') + '''

''' + fig(EXP1_AUC, 3,
          '同一次运行的验证 AUROC，验证子集 62 条视频、4 个身份。首轮 0.4928，末轮 0.4248，最高 0.6039（第 14 轮），全程均值 0.4477；自第 24 个 epoch 起，37 个 epoch 中有 1 个高于 0.50。') + '''

<div class="tw">
<table>
  <caption><b>表 5.</b> 实验一测试集结果，54 真 / 54 假，video-level。阈值在验证集上按 5% 假阳率校准。阶段 B 在第 19 个 epoch 触发早停，验证 AUROC 最佳 0.6627、全程均值 0.6259。</caption>
  <thead><tr><th>指标</th><th class="n">数值</th><th>说明</th></tr></thead>
  <tbody>
    <tr class="head"><td>Video-level AUROC</td><td class="n key">0.4877</td><td>按身份聚类 95% 自助区间 <span class="m">[0.458, 0.531]</span>，4 个身份簇</td></tr>
    <tr><td>Average Precision</td><td class="n">0.4883</td><td>类别平衡，基准率 0.50</td></tr>
    <tr><td>Accuracy</td><td class="n">0.4815</td><td>校准阈值下</td></tr>
    <tr><td>Balanced Accuracy</td><td class="n">0.4815</td><td></td></tr>
    <tr><td>EER</td><td class="n">0.5093</td><td></td></tr>
    <tr><td>TPR @ 5% FPR</td><td class="n">0.0000</td><td></td></tr>
    <tr><td>混淆矩阵</td><td class="n">[[52, 2], [54, 0]]</td><td>[真: TN, FP] / [假: FN, TP]</td></tr>
    <tr><td>真实 / 伪造后验均值</td><td class="n">0.8330 / 0.8478</td><td></td></tr>
  </tbody>
</table>
</div>

''' + fig(EXP1_ROC, 4,
          '实验一测试集，由 108 条逐视频异常分数绘制。<b>(a)</b> 分数分布，真实均值 −0.8330、伪造 −0.8478；黑色虚线为校准阈值。<b>(b)</b> ROC 曲线，AUC 0.4877，按身份聚类 95% 自助区间 [0.458, 0.531]。') + '''

<h2><span class="n">4</span>实验三：CelebDF++ 人脸裁剪输入</h2>

<p><b>目的：</b>在覆盖 22 种伪造方法的 2025 年基准上，测量同一流水线的端到端性能。CelebDF++ 的伪造方法分三个家族：换脸（FaceSwap）、面部重演（FaceReenact）、语音驱动的说话人生成（TalkingFace）。</p>

<div class="tw">
<table>
  <caption><b>表 6.</b> CelebDF++ 划分统计。测试子集的真假比约为 1:32。</caption>
  <thead><tr><th>子集</th><th class="n">真实</th><th class="n">伪造</th><th class="n">身份数</th><th>用途</th></tr></thead>
  <tbody>
    <tr><td>训练</td><td class="n">554</td><td class="n">28150</td><td class="n">245</td><td>阶段 A / 阶段 B</td></tr>
    <tr><td>验证</td><td class="n">166</td><td class="n">5075</td><td class="n">57</td><td>checkpoint 选择</td></tr>
    <tr><td>测试</td><td class="n">170</td><td class="n">5433</td><td class="n">57</td><td>最终评估，全程封存</td></tr>
    <tr><td class="dim">供体约束剔除</td><td class="n dim">0</td><td class="n dim">14538</td><td class="n dim">—</td><td class="dim">不参与任何阶段</td></tr>
  </tbody>
</table>
</div>

<p>人脸检出率：训练集真实 99.3%、伪造 99.2%；验证集 99.2% / 99.4%；测试集 98.1% / 99.1%。检测框宽度中位数：测试集真实 138 px、伪造 156 px。</p>

''' + fig(EXP3_A, 5,
          '实验三阶段 A 的 60 个 epoch。<b>(a)</b> 训练交叉熵自 1.5420 降至 0.5618，最低 0.5279；虚线 ln 2 含义同图 2。<b>(b)</b> 验证 AUROC 自 0.5864 升至 0.6918，最高 0.6937（第 57 轮），全程均值 0.6572。验证子集 5241 条视频、57 个身份。') + '''

''' + fig(EXP3_B, 6,
          '实验三阶段 B 的 17 个 epoch（早停耐心 8）。验证 AUROC 最高 0.6569（第 9 轮），全程均值 0.6198，末轮 0.6255。同一 checkpoint 在测试集上的读数以标记点给出。') + '''

<div class="tw">
<table>
  <caption><b>表 7.</b> 实验三测试集结果，170 真 / 5433 假，其中 3 条因检测缺失或解码失败被跳过，实际计入 169 真 / 5431 假。阈值取自 checkpoint。测试子集基准率为 0.9698。</caption>
  <thead><tr><th>指标</th><th class="n">数值</th><th>说明</th></tr></thead>
  <tbody>
    <tr class="head"><td>Video-level AUROC</td><td class="n key">0.5284</td><td>按身份聚类 95% 自助区间 <span class="m">[0.502, 0.572]</span>，57 个身份簇</td></tr>
    <tr><td>EER</td><td class="n">0.5019</td><td></td></tr>
    <tr><td>Balanced Accuracy</td><td class="n">0.5094</td><td></td></tr>
    <tr><td>TPR @ 5% FPR</td><td class="n">0.0755</td><td></td></tr>
    <tr><td>Accuracy</td><td class="n">0.1041</td><td>基准率 0.9698</td></tr>
    <tr><td>Average Precision</td><td class="n">0.9733</td><td>基准率 0.9698</td></tr>
    <tr><td>混淆矩阵</td><td class="n">[[159, 10], [5007, 424]]</td><td>[真: TN, FP] / [假: FN, TP]</td></tr>
    <tr><td>真实 / 伪造异常分数均值</td><td class="n">−0.8564 / −0.8173</td><td>标准差 0.2716 / 0.2261</td></tr>
    <tr><td>后验预测标准差均值</td><td class="n">0.2086</td><td>16 次蒙特卡洛采样</td></tr>
  </tbody>
</table>
</div>

''' + fig(EXP3_ROC, 7,
          '实验三测试集，由 5600 条逐视频异常分数绘制。<b>(a)</b> 分数分布；因真假比为 1:32，纵轴取密度而非计数。<b>(b)</b> ROC 曲线，AUC 0.5284，由同一份分数文件独立重算，与表 7 一致。') + '''

''' + fig(EXP3_METHOD, 8,
          '实验三测试集 AUROC 按 22 种伪造方法拆分，分三个家族排列。TalkingFace 每种方法 590–625 条伪造视频，FaceReenact 每种 67 条，FaceSwap 每种 69 条（Celeb-DF-v2 为 201 条）；三个家族的真实视频均为同一批 169 条。') + '''

<h2><span class="n">5</span>特征表征测量</h2>

<p><b>目的：</b>阶段 A 交付的是提取器而非分类器，其分类头两层之间不含激活函数，在数学上等价于一个线性泛函。因此直接测量：<b>一个线性分类器能否从提取器输出中区分真伪，且在未见身份上成立</b>。</p>

<p><b>方法：</b>冻结提取器，取其输出特征，用岭回归（对偶闭式解）按身份分五折做交叉验证，报告留出折上的 AUROC。同一测量对随机初始化的提取器重复多次，给出随机参照的取值范围。t-SNE 投影先经 PCA 降至 30 维，其上按身份留一做 5-NN 分类。</p>

''' + fig(TSNE, 9,
          '60 条 DFD 训练视频（30 真 / 30 假，覆盖 20 个身份）提取器特征的 t-SNE 投影。t-SNE 把每条视频的 15488 维特征映射成平面上的一个点，规则是让原本相近的向量在平面上也相近；它只用于观察，不参与任何计算。绿点为真实视频，橙叉为伪造视频。该平面上按身份留一的 5-NN 分类准确率为 0.375。') + '''

<div class="tw">
<table>
  <caption><b>表 8.</b> 留出身份线性探针读数，每次测量 60 条视频。「随机初始化」是同一架构但未经训练的提取器，每行一次抽样；「差值」为两者之差。<b>第一行是量表</b>：它测的是一个端到端测试 AUROC 达 0.7773 的提取器，用于显示该探针会读出多大的增益。<b>该提取器训练于供体约束施加到 CelebDF++ 划分之前</b>，其训练数据包含后来被划入测试集的供体身份；同一配置在供体安全划分上（实验三）不再产生此增益。此行为历史诊断，不作为干净的正对照。</caption>
  <thead><tr><th>测量对象</th><th class="n">checkpoint 轮次</th><th class="n">随机初始化</th><th class="n">训练后</th><th class="n">差值</th></tr></thead>
  <tbody>
    <tr><td class="dim">量表（历史诊断）：一个端到端测试 AUROC 为 0.7773 的提取器</td><td class="n dim">38</td><td class="n dim">0.5778</td><td class="n dim">0.7322</td><td class="n dim">+0.1544</td></tr>
    <tr class="head"><td>实验一 · DFD，整帧</td><td class="n">60</td><td class="n">0.4900</td><td class="n">0.5033</td><td class="n">+0.0133</td></tr>
    <tr class="head"><td>实验三 · CelebDF++，人脸裁剪</td><td class="n">57</td><td class="n">0.7552</td><td class="n">0.7149</td><td class="n">−0.0402</td></tr>
  </tbody>
</table>
</div>


<h2><span class="n">6</span>元数据基线</h2>

<p><b>目的：</b>检查在不解码任何像素的前提下，仅凭容器与编码层面的元信息能把两类分开到什么程度。这类信息不含任何伪造证据，因此它给出的是「数据集本身有多可分」的下界参照。</p>

<p><b>方法：</b>对每条视频取一个标量（画面宽高、时长、文件码率、每像素比特数），按该标量排序，把这个排序当作检测器打分。AUROC 关于方向对称——读数 0.2174 与 0.7826 的判别力相同——因此看的是它离 0.50 的距离，而非方向。区间由按身份聚类的自助重采样给出（2000 次）。</p>

<div class="tw">
<table>
  <caption><b>表 9.</b> 单标量元数据控制项在两个测试子集上的 AUROC 与 95% 身份聚类区间。<b>码率一项在 CelebDF++ 上读 0.2174，等效判别力 0.7826</b>；同一项在 DFD 上为 0.5243。DFD 的四个几何量精确等于 0.5000，因为该数据集两类视频均为 1920×1080。裁剪尺度相关的控制项需要逐视频的人脸框缓存，未包含在本表内。</caption>
  <thead><tr><th>控制项</th><th>CelebDF++ 测试集</th><th>DFD 测试集</th></tr></thead>
  <tbody>
    <tr class="head"><td>文件码率 (MB/s)</td><td>0.2174 &nbsp;<span class="m">[0.187, 0.240]</span></td><td>0.5243 &nbsp;<span class="m">[0.454, 0.616]</span></td></tr>
    <tr class="head"><td>每像素比特数</td><td>0.6108 &nbsp;<span class="m">[0.568, 0.645]</span></td><td>0.5243 &nbsp;<span class="m">[0.454, 0.616]</span></td></tr>
    <tr><td>画面宽度</td><td>0.0759 &nbsp;<span class="m">[0.050, 0.098]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>画面像素数</td><td>0.0775 &nbsp;<span class="m">[0.050, 0.102]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>宽高比</td><td>0.1240 &nbsp;<span class="m">[0.084, 0.168]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>画面高度</td><td>0.3940 &nbsp;<span class="m">[0.377, 0.410]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>帧数</td><td>0.1051 &nbsp;<span class="m">[0.068, 0.138]</span></td><td>0.3923 &nbsp;<span class="m">[0.325, 0.472]</span></td></tr>
  </tbody>
</table>
</div>

<p>为检查模型输出与其中最强的一项之间是否存在单调关联，我们在实验三的同一批 5600 条测试视频上，计算了模型异常分数与码率、每像素比特数的 Spearman 秩相关。</p>

<div class="tw">
<table>
  <caption><b>表 10.</b> 模型异常分数与元数据的 Spearman 秩相关，实验三测试集。同一子集上模型自身的 AUROC 为 0.5284。秩相关只能检出单调关联，无法排除非线性依赖或与其他量的交互。</caption>
  <thead><tr><th>子集</th><th class="n">视频数</th><th class="n">ρ 与码率</th><th class="n">ρ 与每像素比特数</th></tr></thead>
  <tbody>
    <tr><td>全部</td><td class="n">5600</td><td class="n">−0.0218</td><td class="n">−0.0049</td></tr>
    <tr><td>仅真实</td><td class="n">169</td><td class="n">−0.0866</td><td class="n">−0.0890</td></tr>
    <tr><td>仅伪造</td><td class="n">5431</td><td class="n">−0.0192</td><td class="n">−0.0040</td></tr>
  </tbody>
</table>
</div>

<h2><span class="n">7</span>重采样单位对区间的影响</h2>

<p><b>目的：</b>AUROC 的自助区间取决于重采样的单位。一条真实片段在 CelebDF++ 上平均衍生 84 条伪造视频，它们共享背景、服装、光照与镜头运动；按视频重采样把它们当作独立观测，按身份重采样则把它们作为一个整体抽取。本节测量这两种选择在同一批分数上给出的区间差异。</p>

<p><b>方法：</b>对每个实验的逐视频测试分数做两次自助重采样，各 2000 次，随机种子相同。第一次以单条视频为单位，第二次以目标身份为单位。只落到单一类别的抽样被跳过而非计分。</p>

<div class="tw">
<table>
  <caption><b>表 11.</b> 同一批测试分数在两种重采样单位下的 95% 区间。<b>在实验三上，逐视频区间覆盖 0.50，身份聚类区间不覆盖</b>。实验一的身份聚类区间建立在 4 个簇之上，可产生的不同重采样极少，应作为指示性而非精确结果读取。</caption>
  <thead><tr><th>运行</th><th class="n">AUROC</th><th>逐视频 95% 区间</th><th>身份聚类 95% 区间</th><th class="n">身份簇数</th></tr></thead>
  <tbody>
    <tr><td>实验一 · DFD，整帧</td><td class="n">0.4877</td><td><span class="m">[0.376, 0.600]</span> 宽 0.223</td><td><span class="m">[0.458, 0.531]</span> 宽 0.073</td><td class="n">4</td></tr>
    <tr><td>实验三 · CelebDF++，人脸裁剪</td><td class="n">0.5284</td><td><span class="m">[0.484, 0.573]</span> 宽 0.089</td><td><span class="m">[0.502, 0.572]</span> 宽 0.071</td><td class="n">57</td></tr>
  </tbody>
</table>
</div>

<footer>
  数据来源：<span class="m">results/curves.csv</span>、<span class="m">results/summary.csv</span>、<span class="m">results/run_stage_a_dfd_decimate/reports/</span>（实验一 108 条逐视频分数）、<span class="m">results/run_stage_a_celebdfv3_face_stageb_celeb/</span>（实验三指标、22 种方法拆分与 5600 条逐视频分数）、<span class="m">results/run_stage_a_celebdfv3_face_pretrain/history.csv</span>、<span class="m">results/diagnostics/</span>（探针与控制项，含 <span class="m">shortcut_controls_celeb_face.json</span>，见表 9）、<span class="m">scripts/video_size_reports/video_sizes.csv</span>（表 9、表 10）。架构与参数量由 <span class="m">src/video_bcnn/model.py</span> 实例化后读出。完整流水账见 <span class="m">docs/stage_a_experiment_log.md</span>。
</footer>

</div>

''' + SCRIPT + '''
'''

out = REPO / "docs/experiment_report_zh.html"
out.write_text(DOC, encoding="utf-8")
print("wrote", out, len(DOC), "chars")
