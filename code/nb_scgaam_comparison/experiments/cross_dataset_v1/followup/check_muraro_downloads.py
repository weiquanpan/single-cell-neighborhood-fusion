"""Verify supplied author matrix against sealed Muraro, without model evaluation."""
from pathlib import Path
import sys,csv,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import EXP,sha,save,utc
import numpy as np
import pandas as pd
import h5py
from scipy import sparse

def main():
    data=Path('F:/cluster-noise/data');matrix=data/'all_data_Cels2016.csv';annotation=data/'all_data_annotation_Cels2016.csv'
    dest=EXP/'followup/muraro_download_audit';dest.mkdir(exist_ok=True)
    ann=pd.read_csv(annotation,sep='\t',index_col=0)
    with h5py.File(EXP/'heldout_sealed/Muraro.h5','r') as f:
        decode=lambda a:[v.decode() if isinstance(v,bytes) else str(v) for v in a]
        cells=decode(f['obs_names'][...]);genes=decode(f['var_names'][...]);labels=decode(f['obs/cell_type1'][...]);q=f['exprs']
        ref=sparse.csr_matrix((q['data'][...],q['indices'][...],q['indptr'][...]),shape=tuple(q['shape'][...])).tocsc()
    with matrix.open(newline='') as f:columns=next(csv.reader(f,delimiter='\t'))
    assert len(columns)==len(set(columns)) and ann.index.is_unique
    missing_cells=sorted(set(cells)-set(columns));missing_ann=sorted(set(cells)-set(ann.index));assert not missing_cells and not missing_ann
    pos=np.array([columns.index(c) for c in cells]);gene_index={g:i for i,g in enumerate(genes)};seen={};rows=0;noninteger=0;nonzero=0;finite=True;minimum=float('inf');maximum=0.;mismatch=0;max_abs=0.;extras=0
    aligned=np.zeros((len(genes),len(cells)),dtype=np.float64);duplicates={}
    for chunk in pd.read_csv(matrix,sep='\t',index_col=0,chunksize=128,float_precision='round_trip'):
        a=chunk.to_numpy(dtype=np.float64);rows+=len(chunk);finite=finite and bool(np.isfinite(a).all());minimum=min(minimum,float(np.min(a)));maximum=max(maximum,float(np.max(a)));noninteger+=int(np.count_nonzero(a!=np.floor(a)));nonzero+=int(np.count_nonzero(a))
        for i,name in enumerate(chunk.index):
            # Explicit author chromosome suffix removal, with duplicate detection.
            g=str(name).split('__chr',1)[0]
            if g not in gene_index:extras+=1;continue
            if g in seen:duplicates.setdefault(g,[seen[g]]).append(str(name))
            else:seen[g]=str(name)
            aligned[gene_index[g]]+=a[i,pos]
    tolerance_mismatch=0
    for g in seen:
        b=ref[:,gene_index[g]].toarray().ravel();delta=np.abs(aligned[gene_index[g]]-b)
        mismatch+=int(np.count_nonzero(delta));tolerance_mismatch+=int(np.count_nonzero(delta>1e-10));max_abs=max(max_abs,float(delta.max()))
    missing_genes=sorted(set(genes)-set(seen))
    cross={}
    for c,label in zip(cells,labels):
        key=(str(ann.loc[c].iloc[0]),label);cross[key]=cross.get(key,0)+1
    correspondence=[{'author_label':a,'sealed_label':b,'cells':n} for (a,b),n in sorted(cross.items())]
    consistent=all(len({b for a,b in cross if a==label})==1 for label in {a for a,b in cross}) and all(len({a for a,b in cross if b==label})==1 for label in {b for a,b in cross})
    result={'utc':utc(),'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in [matrix,annotation]},'format':'tab-separated despite csv extension; genes rows, cells columns','author_shape_genes_by_cells':[rows,len(columns)],'annotation_rows':len(ann),'all_matrix_cells_have_annotation':set(columns)<=set(ann.index),'sealed_shape_cells_by_genes':list(ref.shape),'missing_cells':missing_cells,'missing_annotations':missing_ann,'missing_genes':missing_genes,'extra_author_genes':extras,'finite':finite,'minimum':minimum,'maximum':maximum,'nonzero_entries':nonzero,'noninteger_entries':noninteger,'aligned_entries_compared':int(ref.shape[0]*len(seen)),'aligned_different_entries':mismatch,'aligned_max_abs_difference':max_abs,'label_correspondence':correspondence,'label_mapping_one_to_one':consistent,'integer_count_compatible':finite and minimum>=0 and noninteger==0,'heldout_clustering_metrics_computed':False}
    result.update(duplicate_gene_symbols=duplicates,duplicate_handling_tested='sum chromosome-specific rows with same symbol; verification only, no input data modified',aligned_difference_gt_1e_10=tolerance_mismatch,initial_check='Rejected duplicate symbol AKAP17A; initial script preserved')
    save(dest/'audit.json',result)
    lines=['# 用户下载的Muraro文件核查','',f'作者矩阵：{rows}基因 × {len(columns)}细胞；注释{len(ann)}行。',f'封存矩阵：{ref.shape[0]}细胞 × {ref.shape[1]}基因。',f'缺失细胞{len(missing_cells)}，缺失注释{len(missing_ann)}，缺失基因{len(missing_genes)}。',f'移除作者基因名的__chr染色体后缀并按ID重排后，比较{result["aligned_entries_compared"]}个值，不同值{mismatch}，最大绝对差{max_abs}。',f'全部表达有限：{finite}；最小值{minimum}；非零值{nonzero}，其中非整数{noninteger}。','','| 作者注释 | 封存注释 | 细胞数 |','|---|---|---:|']
    lines += [f'| {a} | {b} | {n} |' for (a,b),n in sorted(cross.items())]
    lines += ['',f'发现{len(duplicates)}个重复基因符号（染色体后缀不同）；上面的数值比较先对同符号行求和，超过1e-10的差异共{tolerance_mismatch}个。仅用于验证，不改写输入。重复符号明细见audit.json。']
    lines+=['','注释用于来源和对应关系核验，未计算聚类指标或调参。','文件可用于来源/细胞核验；若非整数，则不能直接进入当前冻结的整数计数协议。非整数不代表文件损坏；不对表达取整，不推断饱和端点原始计数。','复现：env313 Python执行 followup/check_muraro_downloads.py。']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(result))

if __name__=='__main__':main()
