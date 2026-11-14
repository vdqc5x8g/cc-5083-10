# 图像 I/O 契约（imageio 组件）

`packages/imageio/` 是焦点融合工程新的可构建读写组件。它把“像素在内存里
长什么样、落盘时哪些能无损表示、哪些必须显式承认损失”写成显式契约，替代
过去散落在 CLI/GUI 里对 `src/core/png.js` 的直接调用。

## 依赖与支持格式（先确认再使用）

- 运行时依赖：**零 npm 依赖**。仅使用标准 `CompressionStream` /
  `DecompressionStream`（Node ≥ 18 与现代浏览器内置）和标准类型化数组。
- 依赖锁定：根 `package-lock.json` 与 `packages/imageio/package-lock.json`
  均已提交，锁文件版本 3，依赖树为空。
- 支持的字节格式（按**魔数签名**分派，不看扩展名）：
  - `PNG`：8/16 位，灰度/RGB/RGBA/调色板/Adam7/tEXt（复用旧编解码器）。
  - `FFI v1`（`FFIMG001`，组件自有容器）：`u8` / `u16` / `u32` / `f32`，
    任意通道数，zlib 压缩，带载荷 CRC-32 与 JSON 元信息。

## 内存契约：ImageTensor

| 字段 | 含义 |
| --- | --- |
| `width`,`height` | 正整数尺寸 |
| `channels` | 正整数，允许任意通道数 |
| `kind` | `u8` / `u16` / `u32` / `f32` |
| `data` | 交错类型化数组，长度 `w*h*channels`；构造器与 `kind` 严格匹配 |
| `channelOrder` | 每通道显式标签（如 `['R','G','B','A','Depth']`），不允许凭通道数猜测 |
| `valueRange` | `{min,max}` 数值域，显式声明而非隐含 |
| `transfer` | `srgb` / `linear` / `unknown` |
| `metadata` | 纯 JSON 可序列化对象 |

## 新旧契约对照

| 维度 | 旧契约（`src/core/png.js` + 调用方） | 新契约（imageio） |
| --- | --- | --- |
| 位深 | 解码一律放大到名义 16 位；8 位文件被 `/255*65535` 缩放，源位深丢失 | 张量恒为 `u16` 承载像素，但 `metadata.sourceBitDepth` 保留源位深；FFI 可保留 `u8/u16/u32` 原生位深 |
| 浮点范围 | 无浮点概念，超出即被夹取/取整 | `f32` 以 IEEE-754 原样存储，不钳制；`valueRange` 记录真实域，HDR（>1.0）可无损往返 |
| 通道顺序 | 恒为 RGBA 4 通道，无法表达额外通道 | `channelOrder` 显式标注，FFI 支持任意通道；进 RGBA 引擎时额外通道默认**拒绝**，显式 `allowLoss` 才丢弃并记录 `dropped-channels` |
| 元信息 | 仅 tEXt 键值，传输曲线靠调用方默认 | FFI 头 JSON 保存 `transfer/valueRange/metadata`；PNG 适配层写入 tEXt 并在 `metadata` 回传源信息 |
| 不可表示内容 | 静默钳到 8/16 位 | PNG 写浮点/降位深默认**抛错**；`allowLoss:true` 才执行，并在返回值 `losses` 与报告中逐项说明（`quantize`/`bit-depth`/`transfer`/`dropped-channels`） |

## 工程描述转换器（project.js）

- 新描述：`focus-fusion-project/v2`，含确定性 `id`（对焦点组成员的
  名称/文件/对焦位置做 FNV 指纹，与临时选项无关）。
- 旧 `manifest.json` / 临时焦点组 → v2：**切片顺序、对焦值、ROI 矩形逐字保留**。
- 纯函数、不改输入；重复转换得到相同 `id` 与相等文档（焦点身份稳定）。
- 磁盘描述的 `id` 与内容不符会被 `parseProject` 拒绝；迁移失败时原工程不动，
  CLI 以临时文件 + rename 原子写出。

## 正式融合路径

CLI（`bin/focus-fusion.js`）、Worker（`src/web/fusion-worker.js`）与 GUI
（`src/web/main.js`）的读写都经由 `readImage` / `writeImage`：

- 读：`readImage(bytes)` → `tensorToSlice(tensor)` 得到引擎 RGBA16 sRGB 切片；
  16 位 sRGB RGBA 零损失进入，其余情形显式报告损失。
- 写：`--format png`（8/16 位，8 位为已确认损失）或 `--format ffi`
  （f32 线性光、真实范围与元信息，零损失）。
