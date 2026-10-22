"""KMLE graphical abstract; saved results, no new model fitting."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
O=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(14,8),facecolor='#f5f8fa');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,14),ylim=(0,8));ax.axis('off')
navy='#15384b';teal='#007e83';blue='#426da1';gray='#587080';orange='#b96c31'
def text(x,y,t,z=14,c=navy,b=False):ax.text(x,y,t,fontsize=z,color=c,weight='bold' if b else 'normal',va='top',linespacing=1.35)
def box(x,y,w,h,fc='white'):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=.12',fc=fc,ec='#d5e1e6'))
def arrow(x,y,u,v):ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=18,lw=1.6,color=gray))
text(.55,7.6,'CDLD representations in the Korean Medical Licensing Examination',22,b=True)
text(.55,7.12,'Information for examinee and item analysis from response correctness',15,c=gray)
box(.55,4.85,4.0,1.8);text(.8,6.4,'KMLE 2017',18,c=blue,b=True);text(.8,5.95,'3,259 examinees × 360 items',14);text(.8,5.52,'1,173,240 binary responses',14)
arrow(4.65,5.75,5.25,5.75);box(5.4,4.85,8.0,1.8,'#e6f2f2');text(5.65,6.4,'CYCLIC DUAL LATENT DISCOVERY',13,c=teal,b=True);text(5.65,5.98,'Student Latent Finder  ↔  Item Latent Finder',16,b=True);text(5.65,5.49,'Alternating updates → frozen 64-dimensional vectors',13)
box(.55,1.25,6.25,3.15);box(7.05,1.25,6.35,3.15)
text(.8,4.14,'2PL ABILITY',15,c=blue,b=True);text(7.3,4.14,'2PL LOG DISCRIMINATION',15,c=teal,b=True)
for x,vals,col in [(.8,[.9289,.9594,.9843],blue),(7.3,[.8759,.8950,.9216],teal)]:
 for y,label,v in zip([3.56,2.98,2.40],['Augmented summaries','+ SVD64','+ CDLD'],vals):
  text(x,y,label,14);text(x+4.2,y,f'{v:.4f}',18,c=col,b=True)
 text(x,1.72,'R² · exploratory nested-CV comparison',11,c=gray)
box(.55,.32,12.85,.64,navy);text(.8,.78,'CDLD adds predictive information beyond augmented summaries and SVD64.',16,c='white',b=True)
for ext in ['png','svg','pdf']:fig.savefig(O/('graphical_abstract.'+ext),dpi=300)
plt.close(fig)
