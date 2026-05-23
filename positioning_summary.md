# 蓝牙定位实验 raw / smoothed 生成总结

这份总结对应当前仓库里的实验版本演进，目标是把 linear、indoor、outdoor 三类实验里 raw 数据和 smoothed 数据是怎么得到的讲清楚。

## 1. 共同的基础拟合

无论是 linear 还是 indoor / outdoor，第一步都先把 RSSI 和距离联系起来。代码里使用的是路径损耗模型：

$$
\mathrm{RSSI}(d) = a - 10 n \log_{10}(d)
$$

反过来把 RSSI 转成距离时，用的是：

$$
\hat d = 10^{\frac{a-r}{10n}}
$$

其中 $r$ 是测到的 RSSI，$a$ 和 $n$ 是通过最小二乘拟合出来的参数。当前实现还做了两层清洗：

- 手工剔除明显异常的点。
- 再按残差阈值迭代剔除离群点，保留足够样本后重新拟合。

因此，后面的 raw / smoothed 不是直接来自原始 RSSI，而是来自“拟合后的路径损耗模型 + 反推距离/位置”的结果。

## 2. Linear 实验

### raw 数据

linear 的 raw 数据是每个距离点对应的估计距离，流程是：

1. 先用上面的路径损耗模型拟合参数 $a, n$。
2. 对每个 RSSI 观测 $r_i$，反推得到估计距离：

$$
\hat d_i = 10^{\frac{a-r_i}{10n}}
$$

3. 为了避免极端值影响图像展示，代码里还会把距离裁剪到 $[0, 26]$。

所以 linear 的 raw 本质上是“RSSI -> 距离”的逐点反演结果。

### smoothed 数据

linear 的 smoothed 是对估计距离做三点加权平均。对中间点 $i$：

$$
\tilde d_i = 0.2\hat d_{i-1} + 0.6\hat d_i + 0.2\hat d_{i+1}
$$

两端点不做平滑，直接保留原值。

### 版本说明

linear 这部分在当前代码里没有像轨迹那样分成不同数学版本；v1 到 v4 的主要变化是图形标注方式，而不是 raw / smoothed 的生成公式。

## 3. Indoor / Outdoor 实验

indoor 和 outdoor 都是三点定位/多点轨迹定位，raw 和 smoothed 的定义在不同版本里变化比较大。

### v1

#### raw 数据

v1 的 raw 是逐点三边定位结果。做法是：

1. 先对每个接收端分别拟合路径损耗参数。
2. 把每个点的 RSSI 反推成三个锚点到发射端的距离。
3. 对单个点单独做非线性最小二乘三边定位，得到该点的二维坐标。

可以写成对第 $i$ 个点求：

$$
\min_{x_i} \sum_k \left(\lVert x_i-a_k\rVert - d_{ik}\right)^2
$$

其中 $a_k$ 是锚点坐标，$d_{ik}$ 是由 RSSI 反推出来的距离。

#### smoothed 数据

v1 的平滑是指数移动平均（EMA）：

$$
s_0 = p_0, \qquad s_i = \alpha p_i + (1-\alpha)s_{i-1}
$$

当前脚本里使用的参数是：

$$
\alpha = 0.45
$$

这里 $p_i$ 是 raw 轨迹点，$s_i$ 是平滑后的轨迹点。

### v2

#### raw 数据

v2 的 raw 仍然是逐点三边定位，和 v1 的 raw 生成方式一致：先 RSSI 反推距离，再对每个点单独做非线性最小二乘定位。

#### smoothed 数据

v2 把 v1 的 EMA 换成了中心三点加权平均：

$$
s_i = 0.2 p_{i-1} + 0.6 p_i + 0.2 p_{i+1}
$$

这个公式对二维坐标是逐维同时应用的，也就是 x、y 坐标分别做同样的加权。

### v3

#### raw 数据

v3 开始不再是单点独立三边定位，而是把整条轨迹一起做全局最小二乘。raw 轨迹对应较小的曲率惩罚权重 $\mu_{raw}$。

优化目标可以写成：

$$
\min_{\{x_i\}} \sum_i \sum_k w_k\left(\lVert x_i-a_k\rVert - d_{ik}\right)^2
 + \mu_{raw} \sum_{i=1}^{N-2} \lVert x_{i+1}-2x_i+x_{i-1}\rVert^2
$$

其中：

- $x_i$ 是第 $i$ 个轨迹点的位置。
- $a_k$ 是第 $k$ 个锚点。
- $d_{ik}$ 是由 RSSI 反推的距离。
- $w_k$ 是由拟合误差倒数构造的权重。
- 第二项是二阶差分曲率惩罚，用来限制轨迹不要过于抖动。

这个“raw”已经不是简单的点对点三边定位，而是带轻微平滑约束的全局轨迹解。

#### smoothed 数据

v3 的 smoothed 也是同一个全局最小二乘框架，只是把曲率惩罚加大，得到更平滑的轨迹：

$$
\min_{\{x_i\}} \sum_i \sum_k w_k\left(\lVert x_i-a_k\rVert - d_{ik}\right)^2
 + \mu_{smooth} \sum_{i=1}^{N-2} \lVert x_{i+1}-2x_i+x_{i-1}\rVert^2
$$

并且当前实现里满足：

$$
\mu_{smooth} > \mu_{raw}
$$

所以 smoothed 比 raw 更“顺”。

### v4

v4 的数学和 v3 一样，raw / smoothed 的生成公式没有变化，只是把轨迹点加了标注，方便观察和对照。也就是说：

- raw 仍然是 v3 那套全局最小二乘的较小曲率版本。
- smoothed 仍然是更大曲率惩罚的版本。

## 4. 一句话对照

- linear：RSSI 反推距离，raw 是逐点估计距离，smoothed 是三点加权平均。
- indoor / outdoor v1-v2：RSSI 反推距离后逐点三边定位，raw 是点位坐标，smoothed 是 EMA 或三点加权平均。
- indoor / outdoor v3-v4：raw 和 smoothed 都来自整条轨迹的全局最小二乘，只是曲率惩罚强度不同。

## 5. 当前仓库里对应的实现位置

- 路径损耗拟合、反推距离、轨迹全局优化：scripts/positioning_analysis.py
- v1 的 EMA 平滑：scripts/label_v1_blue_points.py
- v2 的三点加权平滑：scripts/label_v2_blue_points.py
- 版本说明：docs/positioning_versions.md
