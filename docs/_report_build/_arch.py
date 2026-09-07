# -*- coding: utf-8 -*-
ARCH = '''<figure>
  <div class="figbox">
    <svg viewBox="0 0 720 512" role="img" aria-label="模型架构与两阶段流程：三段 3D 卷积的提取器输出 15488 维特征；阶段 A 以确定性头监督训练该提取器，训练完成后把提取器权重冻结交给阶段 B，阶段 B 只训练贝叶斯头并输出异常分数。">
      <defs>
        <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
        </marker>
        <marker id="ahk" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--s2)"/>
        </marker>
      </defs>

      <text x="20" y="24" font-family="Source Serif 4, serif" font-size="12.5" font-weight="600" fill="currentColor">特征提取器</text>
      <text x="700" y="24" font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--faint)" text-anchor="end">阶段 A 训练它 · 阶段 B 冻结它</text>

      <g fill="none" stroke="currentColor" stroke-width="1.2" opacity="0.85">
        <rect x="20" y="36" width="88" height="58" rx="3"/>
        <rect x="120" y="36" width="88" height="58" rx="3"/>
        <rect x="220" y="36" width="88" height="58" rx="3"/>
        <rect x="320" y="36" width="88" height="58" rx="3"/>
        <rect x="420" y="36" width="88" height="58" rx="3"/>
        <rect x="520" y="36" width="88" height="58" rx="3"/>
        <rect x="620" y="36" width="88" height="58" rx="3"/>
      </g>
      <g font-family="Source Serif 4, serif" font-size="11.5" fill="currentColor" text-anchor="middle">
        <text x="64" y="59">输入片段</text><text x="164" y="59">卷积段 1</text><text x="264" y="59">卷积段 2</text>
        <text x="364" y="59">卷积段 3</text><text x="464" y="59">时间维均值</text><text x="564" y="59">自适应池化</text>
        <text x="664" y="59">特征向量</text>
      </g>
      <g font-family="IBM Plex Mono, monospace" font-size="9" fill="var(--muted)" text-anchor="middle">
        <text x="64" y="76">3×8×256×256</text><text x="164" y="76">16×8×125×125</text><text x="264" y="76">24×8×59×59</text>
        <text x="364" y="76">32×8×26×26</text><text x="464" y="76">32×26×26</text><text x="564" y="76">32×22×22</text>
        <text x="664" y="76">15488</text>
      </g>
      <g font-family="IBM Plex Mono, monospace" font-size="8.5" fill="var(--faint)" text-anchor="middle">
        <text x="64" y="89">人脸裁剪</text><text x="164" y="89">3→16</text><text x="264" y="89">16→24</text>
        <text x="364" y="89">24→32</text><text x="464" y="89">8 帧取平均</text><text x="564" y="89">+BatchNorm2d</text>
        <text x="664" y="89">展平</text>
      </g>
      <g stroke="currentColor" stroke-width="1.2" marker-end="url(#ah)" opacity="0.7">
        <line x1="108" y1="65" x2="118" y2="65"/><line x1="208" y1="65" x2="218" y2="65"/>
        <line x1="308" y1="65" x2="318" y2="65"/><line x1="408" y1="65" x2="418" y2="65"/>
        <line x1="508" y1="65" x2="518" y2="65"/><line x1="608" y1="65" x2="618" y2="65"/>
      </g>
      <path d="M 120 104 L 120 112 L 408 112 L 408 104" fill="none" stroke="var(--faint)" stroke-width="1"/>
      <text x="264" y="128" font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--faint)" text-anchor="middle">每段：Conv3d k=3×5×5 → AvgPool3d 4×4 步长 2 → BatchNorm3d → ReLU</text>

      <path d="M 664 94 L 664 146 L 366 146" fill="none" stroke="currentColor" stroke-width="1.2" opacity="0.7"/>
      <line x1="366" y1="146" x2="366" y2="168" stroke="currentColor" stroke-width="1.2" marker-end="url(#ah)" opacity="0.7"/>

      <rect x="60" y="172" width="600" height="116" rx="3" fill="none" stroke="var(--s1)" stroke-width="1.4"/>
      <text x="78" y="194" font-family="Source Serif 4, serif" font-size="12.5" font-weight="600" fill="var(--s1)">阶段 A · 监督预训练</text>
      <text x="642" y="194" font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--s1)" text-anchor="end">提取器可训练</text>
      <g font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor">
        <text x="78" y="220">Linear  15488 → 512</text>
        <text x="78" y="238">Dropout p = 0.2</text>
        <text x="78" y="256">Linear  512 → 1</text>
      </g>
      <text x="78" y="277" font-family="Source Serif 4, serif" font-size="10.5" fill="var(--faint)">两层之间无激活函数 · 7,930,881 参数</text>
      <g font-family="Source Serif 4, serif" font-size="11" fill="var(--muted)">
        <text x="330" y="220">目标：二元交叉熵，真伪两类标签</text>
        <text x="330" y="238">Adam，lr 1×10⁻⁴，指数衰减 γ = 0.95</text>
        <text x="330" y="256">每类每 epoch 采样 1000 个片段</text>
        <text x="330" y="277">选取验证 AUROC 最高的一轮</text>
      </g>

      <line x1="366" y1="288" x2="366" y2="334" stroke="var(--s2)" stroke-width="2" marker-end="url(#ahk)"/>
      <text x="382" y="303" font-family="Source Serif 4, serif" font-size="11.5" font-weight="600" fill="var(--s2)">保留提取器权重并冻结，含 BatchNorm 统计量</text>
      <text x="382" y="320" font-family="Source Serif 4, serif" font-size="11" fill="var(--muted)">丢弃阶段 A 的头</text>
      <text x="350" y="303" font-family="Source Serif 4, serif" font-size="11" fill="var(--faint)" text-anchor="end">交付物</text>

      <rect x="60" y="338" width="600" height="116" rx="3" fill="none" stroke="var(--s3)" stroke-width="1.4"/>
      <text x="78" y="360" font-family="Source Serif 4, serif" font-size="12.5" font-weight="600" fill="var(--s3)">阶段 B · 单类异常检测</text>
      <text x="642" y="360" font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--s3)" text-anchor="end">提取器冻结</text>
      <g font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor">
        <text x="78" y="386">FC1 ~ N(μ, σ)  15488 → 512</text>
        <text x="78" y="404">Dropout p = 0.2</text>
        <text x="78" y="422">FC2 ~ N(μ, σ)  512 → 1</text>
      </g>
      <text x="78" y="443" font-family="Source Serif 4, serif" font-size="10.5" fill="var(--faint)">先验 N(0, 0.1²) · KL 权重 10⁻³</text>
      <g font-family="Source Serif 4, serif" font-size="11" fill="var(--muted)">
        <text x="330" y="386">目标：TraceGraph ELBO，Pyro SVI</text>
        <text x="330" y="404">SGD，仅使用真实视频训练</text>
        <text x="330" y="422">异常分数 = 后验预测的负值</text>
        <text x="330" y="443">选取验证 macro AUROC 最高的一轮</text>
      </g>

      <line x1="366" y1="454" x2="366" y2="480" stroke="currentColor" stroke-width="1.2" marker-end="url(#ah)" opacity="0.7"/>
      <text x="366" y="500" font-family="Source Serif 4, serif" font-size="11.5" fill="currentColor" text-anchor="middle">视频级异常分数 → AUROC · EER · TPR@5%FPR</text>
    </svg>
  </div>
  <figcaption><b>图 1.</b> 网络结构与两阶段流程，自上而下依次执行。特征提取器为三段 3D 卷积，每段的顺序是卷积、平均池化、批归一化、激活；三段之后沿时间轴取均值，自适应平均池化到 22×22，再经一次 BatchNorm2d 展平为 15488 维。张量尺寸以人脸裁剪输入 256×256 为例，整帧输入 540×960 经同一路径同样得到 15488 维，尺寸差异由自适应池化吸收。<b>阶段 A</b> 用真伪标签训练提取器与一个确定性头；训练结束后<b>丢弃该头，把提取器权重冻结交给阶段 B</b>；阶段 B 只在真实视频上训练形状相同的均值场贝叶斯头，输出视频级异常分数。</figcaption>
</figure>'''
