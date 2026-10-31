"""Write evidence-based review and archive only after completed verification."""
from pathlib import Path
import json,hashlib,zipfile,decimal
ROOT=Path(__file__).resolve().parent
def load(n):return json.loads((ROOT/n).read_text(encoding='utf8'))
v=load('completion_verification.json');a=load('capture_analysis.json');r=load('result.json')
assert v['historical_printed_epoch_losses_all_exact'] and v['terminal_model_and_optimizer_bitwise_equal_to_original']
row=a['contrast_path']['rows'][0]
decimal.getcontext().prec=40
ratio=decimal.Decimal.from_float(row['log_denominator_gradient'])/decimal.Decimal.from_float(row['denominator'])
minutes=r['elapsed_seconds']/60
report=f'''scSAMAC / Spleen seed101 带诊断日志复现：审查报告
完成时间：{r['finished_utc']}
状态：同配置失败已复现，首个非有限计算路径已定位。

结论

本次复现定位到冻结作者实现的对比损失反向计算：第 74 轮第 25 批次、第 2,799 次更新的反向传播中，log 的输入仍是有限的极小正数，但其梯度计算溢出为 Inf；随后掩码/指数链上的乘法产生 NaN，污染编码器和注意力参数。NB 重建分支与均值激活在触发批次的前向和已观测反向张量均有限。

这比此前只读审计更具体：本次不是 log(0)，也不是均值激活的 exp 溢出，而是“次正规正分母 → log 反向除法溢出 → NaN 传播”。此前审计报告和原失败记录均保留，不将旧推测改写成当时已经确定的结论。

1. 复现范围与一致性

• 仅启动一次诊断训练，seed101，原 Spleen 9,552×2,000 输入。
• 原架构、float64、sigma=2.5、温度 0.07、对比权重 0.01、Adam lr=1e-4、amsgrad=True、batch256 均保持。
• 执行全部 300 轮预训练、11,400 次 Adam 更新；每轮核实 9,552 个细胞各出现一次。
• 300 轮打印的总损失全部与历史 seed101 日志一致，包括前 73 轮的有限值和第 74 轮起的 NaN。历史日志保留八位小数，因此这是日志精度上的一致，不冒充所有历史中间状态逐比特一致。
• 末轮模型和优化器张量按字节与原失败 checkpoint 对比一致（包含 NaN 的位模式）。随后同样在联合训练初始化的嵌入有限性检查处失败，无联合阶段更新。
• 本次诊断耗时 {minutes:.2f} 分钟；逐批次观测和快照复制有明显开销，不能当作该方法的正式训练耗时。新增 11,400 次诊断更新单独记录，不替换原运行成本。
• 原始输入、代码和原 seed101 结果通过前后哈希核查；上一次审计保护的 68 个文件（包括全部五个失败记录的相关文件）也保持原 SHA256。

2. 第一次失败前后的直接证据

位置                 NB 重建损失          对比损失（未乘 0.01）  总损失              梯度/参数
74轮24批 / 步2798     0.18285757107124445  0.5466554911433551     0.188324125982678   观测梯度和更新后参数均有限
74轮25批 / 步2799     0.18191275008748914  2.2389409934896465     0.2043021600223856  前向有限；对比反向先 Inf 后 NaN
74轮26批 / 步2800     NaN                  NaN                    NaN                编码器前向即已 NaN
74轮27批 / 步2801     NaN                  NaN                    NaN                非有限状态继续传播

第 2,799 步开始时的模型和优化器，与第 2,798 步结束时快照逐张量相同，模型参数全部有限。第 2,799 步的对比反向异常之后，22 个编码器/注意力参数张量的梯度出现非有限值，更新后这些参数张量受到污染；此时解码器相关梯度仍有限。后续批次输入的仍是有限的原数据，但模型已经损坏，因此 NB 和对比损失随后一起变为 NaN。

原日志的“NB loss”是 NB + 0.01×对比损失的总和。不能将第 74 轮日志里的 NB loss=nan 解释为 NB 分支先坏。

3. 精确触发运算

冻结源码：source_original/st_loss.py 第 57–80 行；关键 log 位于第 77 行。
原运算为：
  scores = features @ features.T / 0.07
  logits = scores - 每行最大值（包含自相似对角线）
  S = sum(exp(logits) * 排除自相似的掩码)
  log_prob = logits - log(S)

触发行：对比矩阵第 354 行（0 基索引），对应批内位置 98、prepared 输入行 560（0 基索引）、第二个视图。
• 自相似 score：4464.934892692757。
• 最大非自相似 score：3726.512045057574，恰为另一视图的正样本。
• 两者差：738.4228476351827。
• 该行 511 个非自相似指数项中，510 个已下溢为零。
• 剩余的分母 S 约为 2.026×10^-321，仍为有限正数；本批没有零分母。
• log(S) = -738.4239147616829，仍有限。
• 到达 log(S) 的上游梯度 g = 0.01 / 512 = 1.953125×10^-5，仍有限。
• log 反向执行 g/S，数学量级约 {ratio:.6E}，超过 float64 最大有限值约 1.798×10^308，实际保存的梯度为 +Inf。

首次非有限梯度的观测顺序为：
contrast.denominator 的一个 +Inf
→ contrast.masked_exp 该行 512 个 +Inf
→ contrast.exp_values 的梯度在掩码对角线出现 0×Inf=NaN
→ 指数反向中，已下溢为零的项继续产生 0×Inf=NaN
→ logits、特征、编码器/注意力梯度受污染。

这是原计算表达式在有限精度反向传播中的失败。数学上相关指数与对数导数可以抵消，但按原有中间量顺序计算时，先出现的 Inf/NaN 已破坏这种抵消。此结论来自本次实际失败批次保存的张量和梯度，不是仅凭合成输入推测。

4. 不是本次触发源的路径

• 全部已观测前向张量在第 74 轮第 25 批次仍有限。
• NB 分项损失、NB 对均值和离散度的梯度均有限。
• mean.pre_exp 范围为 [-14.99235, 5.28986]，exp 最大值约 198.31554；其前向和反向均有限，未发生此前讨论的均值激活 exp 溢出。
• KMeans 与最终可视化尚未运行，不是这次触发原因。
• 第一次非有限事件之前，没有有限性修复、nan_to_num、梯度裁剪、损失替换或样本跳过。

5. 数据预处理与适配问题的判断边界

此前审计已验证原计数、索引、共享 HVG、size factor 和标准化复算；同一基因集合的作者原始归一化与统一输入仅有浮点精度量级差异。此次故障批次的输入、前向输出和更新前模型均有限，异常明确产生在冻结作者对比损失的 log 反向路径。

因此，本次证据支持：直接数值触发点是原作者实现中的对比损失计算稳定性问题，没有发现输入中含非法 NaN、计数错位或适配器改坏损失公式的证据。

同时，数据分布、共享 HVG、输入精度和运行环境仍是实验条件，可能影响是否以及何时走到危险数值范围。触发行的两个潜在特征范数约为 17.679 和 15.213，输入行范数约 379.558。不能因找到了此处的数值路径，就宣称所有原生输入、软件版本或随机种子必然失败。此次只有 seed101 做了带诊断复现；其它四个 seed 的原失败证据保留，未逐个重新诊断。

6. 可用于论文/审稿的表述

“在原参数和 seed101 下进行独立诊断复现后，scSAMAC 在 Spleen 的第 74 个预训练轮次、第 25 个批次出现对比损失的反向数值溢出。该批次的 NB、对比项及总前向损失均有限，但掩码后的对比分母下降至约 2.026×10^-321，使 log 的反向梯度溢出为 Inf，并进一步产生 NaN。下一批次开始，损失和嵌入受到污染。300 轮总损失日志与原运行一致，诊断保留了失败前后模型、优化器、随机状态与梯度。该结果限定于本文共享输入和软件环境；未通过改参数、换 seed 或替换算法获得结果。”

这可以解释 Figure S2 的缺失面板，并将“原因未明的失败”更新为有直接证据的数值失败。仍不能把缺失单元当作 ARI=0，不能据此证明 Fusion 普遍更稳定，也不能计算 scSAMAC 的完整五数据集排名。

本任务没有修复算法，没有写回原训练目录，没有修改论文结果表或替换失败图。若之后处理数值稳定性，应使用另一个审计版本单列验证与结果，不静默替换本次原实现失败。

7. 证据文件

• first_nonfinite.json：首个非有限事件。
• first_nonfinite_batch.json、last_finite_batch.json、after_trigger_1/2_batch.json：前后分项损失、张量/梯度统计和传播顺序。
• capture_analysis.json：从保存的张量与快照提取的关键数值、因果路径核对。
• checkpoints/first_nonfinite_tensors.pt：实际触发批次的中间张量、参数梯度及已保留中间梯度。
• checkpoints/last_finite_before.pt、last_finite_after.pt、first_nonfinite_before.pt、first_nonfinite_after.pt：异常前后完整模型和优化器；before 文件另含实际批次及 Python/NumPy/CPU/CUDA RNG。
• checkpoints/after_trigger_1/2_before/after.pt：随后两个批次的状态。
• batches.jsonl：全部 11,400 批次的采样位置和分项损失，事件前后附详细观测。
• epoch_trace_comparison.json、completion_verification.json：日志一致性、完整预算、采样覆盖、末轮状态及原文件哈希核验。
• source_original/、source_instrumented/、三个 .diff：冻结实现与纯观测改动。

快照支持实际批次的重演；DataLoader 迭代器本身未序列化，不能把这些文件误当成通用中途续训功能。诊断没有读取标签。
'''
(ROOT/'REPORT_ZH.txt').write_text(report,encoding='utf8')
english='''Diagnostic rerun of scSAMAC on Spleen (seed101)

An independent instrumented rerun retained the original inputs, seed101,
architecture, hyperparameters, and all 300 pretraining epochs (11,400 Adam
updates). Every epoch's aggregate loss matched the archived run at its recorded
eight-decimal precision, and the terminal model and optimizer tensors matched
the archived checkpoint byte for byte, including NaN bit patterns.

The first nonfinite event occurred during backpropagation at epoch 74, batch 25
(update 2,799). The NB loss (0.1819127501), unweighted contrastive loss
(2.2389409935), and total forward loss (0.2043021600) were still finite.
In one contrastive row, subtracting the maximum including the self-similarity
and then masking the diagonal produced a positive subnormal denominator of
approximately 2.026e-321. The incoming gradient to its logarithm was
1.953125e-5, so the logarithm's backward division overflowed to +Inf. Subsequent
mask and exponential multiplications generated NaNs, contaminating encoder
and attention gradients and parameters. Both losses became NaN in batch 26.
The denominator had not reached zero, and the NB branch and mean activation
remained finite at the triggering batch.

These saved-tensor observations locate a numerical instability in the frozen
author contrastive-loss implementation under the stated shared-input and
software conditions. They do not establish that the method fails under all
native preprocessing pipelines, environments, or seeds. No algorithm fix,
parameter change, seed replacement, or overwrite of the original failure was
performed. This missing result is not assigned a zero score or used to support
a general stability advantage for Fusion. Failure-adjacent model/optimizer/RNG
states, gradients, input indices, source diffs, and logs are retained separately.
'''
(ROOT/'REVIEWER_NOTE_EN.txt').write_text(english,encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.zip','.lock'] and p.name not in ['evidence_manifest.json','package_receipt.json'])
manifest={str(p.relative_to(ROOT)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
(ROOT/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
files.append(ROOT/'evidence_manifest.json')
dest=ROOT/'scSAMAC_Spleen_seed101_diagnostic_v1.zip'
with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
    for p in files:z.write(p,str(p.relative_to(ROOT)))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for name,entry in manifest.items():
        h=hashlib.sha256()
        with z.open(name) as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        assert h.hexdigest()==entry['sha256'],name
receipt=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),entries=len(files),all_entry_sha256_and_crc_verified=True,contains_checkpoints=True)
(ROOT/'package_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps(receipt,indent=2))
