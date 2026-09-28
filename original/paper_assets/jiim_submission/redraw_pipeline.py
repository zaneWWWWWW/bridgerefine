import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC']; plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(12,4)); ax.set_xlim(0,12); ax.set_ylim(0,4); ax.axis('off')
def box(x,y,w,h,t,fc,ec):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.12',fc=fc,ec=ec,lw=2); ax.add_patch(p); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=12,wrap=True)
def arrow(x1,y1,x2,y2,c='#334155',style='-|>'):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=18,lw=2,color=c))
ax.text(.3,3.65,'BridgeRefine：冻结扩散桥辅助的 CT→MRI 患者级重建',fontsize=16,fontweight='bold',color='#163B65')
box(.35,1.55,1.65,1.05,'非增强 CT\n$y$', '#E8F1FA','#24558A'); arrow(2.05,2.08,2.55,2.08)
box(2.6,1.45,2.25,1.25,'SelfRDB 扩散桥\n冻结参数\n10 步 × 2 递归','#EEF2F7','#64748B'); arrow(4.95,2.08,5.45,2.08)
box(5.5,1.55,1.75,1.05,'粗略 MRI\n$\hat{x}_{bridge}$','#E7F5F2','#16877B'); arrow(7.35,2.08,7.85,2.08)
box(7.9,1.35,2.15,1.45,'监督精炼器 $R_\theta$\nCT + 粗略 MRI\nL1 损失','#EAF6EE','#248A57'); arrow(10.15,2.08,10.65,2.08,'#248A57')
box(10.7,1.55,1.0,1.05,'合成 MRI\n$\hat{x}$','#DDF1EC','#16877B')
ax.plot([1.2,1.2,8.75],[1.5,.75,.75],color='#24558A',lw=2)
arrow(8.75,.75,8.75,1.3,'#24558A')
ax.text(4.8,.95,'CT 跳跃条件（保留源模态结构）',ha='center',fontsize=10,color='#24558A')
ax.text(8.8,.45,'训练阶段：与配对真实 MRI 计算 L1 损失',ha='center',fontsize=10,color='#9A4A2F')
arrow(11.2,1.5,11.2,.75,'#9A4A2F'); box(10.4,.05,1.7,.5,'配对真实 MRI $x_0$','#FFF1EA','#9A4A2F')
fig.savefig('fig1_pipeline_redrawn.png',dpi=400,bbox_inches='tight'); fig.savefig('fig1_pipeline_redrawn.pdf',bbox_inches='tight'); plt.close(fig)
