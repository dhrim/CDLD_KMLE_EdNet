"""Graphical abstract from submitted English v10, Tables 1–3; no new analysis.
Run: python3 20261004_graphical_abstract.py (requires matplotlib).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig=plt.figure(figsize=(16,10),facecolor='#F6F8FB')
a=fig.add_axes([0,0,1,1]); a.set(xlim=(0,16),ylim=(0,10)); a.axis('off')
navy='#193249'; teal='#007F83'; orange='#AE6224'; muted='#586B7D'
def text(x,y,s,size=14,color=navy,weight='normal',ha='left'):
    a.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='top',linespacing=1.45)
def box(x,y,w,h,fc='white',ec='#DBE3EC'):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.15',facecolor=fc,edgecolor=ec,lw=1))
def arrow(x,y,xx,yy):
    a.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=16,lw=1.7,color=muted,connectionstyle='arc3'))
text(.65,9.58,'CDLD: additional information in learned representations',25,weight='bold')
text(.65,9.05,'Secondary analysis of Korean Medical Licensing Examination (KMLE) and EdNet response data',13,color=muted)
box(.65,6.88,5.0,1.72)
text(.9,8.38,'CORRECT / INCORRECT RESPONSES',11,color=teal,weight='bold')
text(.9,7.98,'KMLE',17,weight='bold'); text(2.05,7.96,'3,259 examinees × 360 items',13)
text(.9,7.49,'EdNet',17,weight='bold'); text(2.05,7.47,'79.9 million response records',13)
arrow(5.8,7.73,6.43,7.73)
box(6.58,6.88,8.77,1.72,fc='#EAF3F5',ec='#C7DFE1')
text(6.86,8.38,'CYCLIC DUAL LATENT DISCOVERY (CDLD)',11,color=teal,weight='bold')
text(6.86,7.98,'Student Latent Finder  ↔  Item Latent Finder',17,weight='bold')
text(6.86,7.48,'Alternating updates → freeze 64-dimensional student and item vectors',12)
arrow(10.9,6.86,4.45,6.3); arrow(10.9,6.86,11.75,6.3)
box(.65,1.62,7.16,4.6)
box(8.08,1.62,7.27,4.6)
text(.93,5.96,'01   IRT ESTIMATE PREDICTION',13,color=teal,weight='bold')
text(.93,5.48,'Information beyond augmented response summaries',13,weight='bold')
text(.93,5.06,'Exploratory comparison · nested cross-validation',11,color=muted)
text(.93,4.55,'KMLE · 2PL log discrimination',14,weight='bold')
x0,x1=4.45,7.13
for v in [.85,.90,.95]:
    x=x0+(v-.85)/.10*(x1-x0)
    a.plot([x,x],[2.96,4.12],color='#E2E8EF',lw=1,zorder=1)
    text(x,2.83,f'{v:.2f}',10,color=muted,ha='center')
for y,label,value,c in [(3.99,'Augmented summaries',.8759,muted),(3.56,'Summaries + SVD64',.8950,'#7490AE'),(3.13,'Summaries + CDLD',.9216,teal)]:
    text(.96,y+.09,label,11,color=c)
    x=x0+(value-.85)/.10*(x1-x0)
    a.scatter([x],[y],s=76,color=c,zorder=3)
    text(x+.11,y+.09,f'{value:.4f}',11,color=c,weight='bold')
text(7.4,4.40,'R²',11,ha='right')
text(.94,2.35,'EdNet · 2PL log discrimination',13,weight='bold')
text(.94,1.99,'R²: 0.6727 → 0.8923 with CDLD added',14,color=teal,weight='bold')
text(8.36,5.96,'02   ITEM-CATEGORY CLASSIFICATION',13,color=teal,weight='bold')
text(8.36,5.48,'EdNet · seven item categories (part)',14,weight='bold')
text(8.36,5.06,'Category labels were not used to learn CDLD vectors',11,color=muted)
text(8.36,4.54,'IRT + response summaries',12,color=muted)
text(13.95,4.54,'+ CDLD',12,color=teal,weight='bold',ha='center')
arrow(12.06,4.43,12.94,4.43)
text(8.36,3.93,'Log loss ↓',12,weight='bold')
text(11.48,3.95,'1.5384',20,color=muted,ha='center')
text(13.95,3.95,'1.2746',20,color=teal,weight='bold',ha='center')
text(8.36,3.21,'Balanced accuracy ↑',12,weight='bold')
text(11.48,3.23,'0.1844',20,color=muted,ha='center')
text(13.95,3.23,'0.3413',20,color=teal,weight='bold',ha='center')
text(8.36,2.32,'Additional item-category information',14,color=teal,weight='bold')
text(8.36,1.96,'beyond IRT estimates and response summaries',12,color=teal)
box(.65,.60,14.7,.72,fc=navy,ec=navy)
text(8,1.08,'Correctness-trained representations add information for examinee and item analysis.',17,color='white',weight='bold',ha='center')
text(.7,.39,'IRT = item response theory; 2PL = two-parameter logistic model; SVD64 = 64-dimensional singular value decomposition representation.',9,color=muted)
fig.savefig(OUT/'20261004_graphical_abstract.png',dpi=300)
fig.savefig(OUT/'20261004_graphical_abstract.svg')
fig.savefig(OUT/'20261004_graphical_abstract_preview.png',dpi=110)
