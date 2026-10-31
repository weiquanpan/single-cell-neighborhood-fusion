from pathlib import Path
import fitz, json, hashlib, re
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent
qa=OUT/'audit/rendered';qa.mkdir(exist_ok=True)
checks=[]
for part in ['manuscript_en','appendix_en']:
    folder=qa/part;folder.mkdir(exist_ok=True)
    doc=fitz.open(OUT/part/'main.pdf');pages=[];text=[];bounds=[]
    for i,p in enumerate(doc):
        pix=p.get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False)
        file=folder/f'page_{i+1:02d}.png';pix.save(file);pages.append(file)
        text.append(p.get_text())
        words=p.get_text('words')
        bad=[w for w in words if w[0]<20 or w[2]>p.rect.width-20 or w[1]<15 or w[3]>p.rect.height-15]
        if bad:bounds.append({'page':i+1,'words':bad[:8]})
    (folder/'extracted_text.txt').write_text('\n\n'.join(text),encoding='utf-8')
    for group in range(0,len(pages),6):
        canvas=Image.new('RGB',(1200,1160),'#d9dee3');draw=ImageDraw.Draw(canvas)
        for j,src in enumerate(pages[group:group+6]):
            im=Image.open(src);im.thumbnail((380,535));x=(j%3)*400+(400-im.width)//2;y=(j//3)*580+28
            canvas.paste(im,(x,y));draw.text((j%3*400+15,j//3*580+8),f'{part} / page {group+j+1}',fill='black')
        canvas.save(folder/f'contact_{group//6+1}.png')
    log=(OUT/part/'main.log').read_text(errors='replace')
    issues=[line for line in log.splitlines() if any(k in line for k in ['Overfull','undefined','Missing character'])]
    checks.append({'project':part,'pages':len(doc),'compiler_issues':issues,'out_of_page_words':bounds,'unresolved_pdf_references':'??' in '\n'.join(text)})
    print(checks[-1])
originals=json.loads((OUT/'audit/original_hashes.json').read_text())
unchanged=all(hashlib.sha256((OUT.parent/'en_revise'/p).read_bytes()).hexdigest()==h for p,h in originals.items())
assert unchanged
(OUT/'audit/pdf_checks.json').write_text(json.dumps({'pdfs':checks,'originals_unchanged':unchanged},indent=2))
