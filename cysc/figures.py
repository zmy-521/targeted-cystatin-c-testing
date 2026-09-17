"""Frozen figure layouts: Python, white background, 300 dpi PNG and TIFF only.

Figure 2: development/model selection, 2x3 grid. Figure 3: raw external ROC/PR/DCA
and cross-fitted local calibration, 2x2 grid. Figure 4: final-panel global SHAP,
bar/beeswarm above nine dependence plots. Figure 5: fixed-budget testing, 2x2.
Figure 6: related renal phenotypes, four groups and shared-scale heatmaps.
S1: pre-redundancy Spearman; S2: seven-model PR; S3: raw external calibration.
No scientific selection or fit occurs in this module except descriptive LOWESS.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import seaborn as sns
import shap
from scipy.special import logit,expit
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix
from .io import FROZEN,output,arrays,load_private,save_table,require,checkpoint
from .evaluation import GROUPS

COLORS={'XGBoost':'#2ca02c','RF':'#1f77b4','LightGBM':'#bcbd22','SVM':'#9467bd','KNN':'#8c564b','LR':'#17becf','DT':'#ff7f0e'}
DISPLAY={'RF':'Random Forest','LR':'Logistic Regression','DT':'Decision Tree','KNN':'K-nearest Neighbors','SVM':'Support Vector Machine'}
COHORTS={'development':'Development cohort','external':'External validation cohort'}
CC={'development':'#2879ae','external':'#d47b32'}

def table(cfg,name):return pd.read_csv(output(cfg)/'tables'/(name+'.csv'))

def style():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
        'font.size':10,'axes.labelsize':11,'axes.labelweight':'bold','axes.linewidth':1,
        'axes.spines.top':False,'axes.spines.right':False,'xtick.direction':'in','ytick.direction':'in',
        'legend.fontsize':9,'legend.frameon':False,'figure.facecolor':'white','axes.facecolor':'white','path.simplify':False})

def save(cfg,fig,name):
    p=output(cfg)/'figures';p.mkdir(exist_ok=True)
    for ext in ['png','tiff']:
        extra={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
        fig.savefig(p/(name+'.'+ext),dpi=300,facecolor='white',**extra)
    plt.close(fig)

def letters(fig,positions):
    for label,x,y in positions:fig.text(x,y,label,fontsize=25,va='top')

def bees(ax,z,max_display=20,order=None):
    names=[{'age':'Age','sex':'Sex','eGFRcr_2021':'eGFRcr'}.get(f,f) for f in z['features']]
    e=shap.Explanation(values=z['values'],data=z['inputs'],feature_names=names)
    np.random.seed(42)
    kwargs={} if order is None else {'order':order}
    shap.plots.beeswarm(e,max_display=max_display,ax=ax,plot_size=None,color_bar=False,
                        group_remaining_features=False,s=4 if max_display>9 else 9,alpha=.85,show=False,**kwargs)
    ax.set_xlabel('SHAP value (impact on model output)',fontsize=10)
    pos=ax.get_position();ca=ax.figure.add_axes([pos.x1+.012,pos.y0,.004,pos.height])
    cb=ax.figure.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=shap.plots.colors.red_blue),cax=ca)
    cb.set_ticks([0,1],labels=['Low','High']);cb.set_label('Feature value',fontsize=9);cb.outline.set_visible(False);cb.ax.tick_params(length=0)

def figure2(cfg):
    style();dev=load_private(cfg,'full_oof');final=load_private(cfg,'final_development_oof')
    met=table(cfg,'Full_model_comparison');red=table(cfg,'Feature_reduction');oo=load_private(cfg,'XGBoost_reduction_oof')
    fig=plt.figure(figsize=(18,9));pos={'A':[.052,.55,.267,.405],'B':[.404,.55,.22,.405],'C':[.742,.55,.22,.405],'D':[.052,.075,.267,.375],'E':[.40,.075,.247,.375],'F':[.728,.075,.267,.375]}
    axes={k:fig.add_axes(v) for k,v in pos.items()}
    for _,r in met[met.Model!='Baseline'].sort_values('AUPRC',ascending=False).iterrows():
        n=r.Model;x,y,_=roc_curve(dev.D2,dev[n]);axes['A'].plot(x,y,color=COLORS[n],lw=1.25,label=f'{DISPLAY.get(n,n)} (AUROC = {r.AUROC:.3f})')
    ax=axes['A'];ax.plot([0,1],[0,1],'--',color='black',lw=1);ax.set(xlim=(0,1),ylim=(0,1.015),xlabel='False Positive Rate',ylabel='True Positive Rate');ax.legend(loc='lower right',fontsize=8)
    for k,n in [('B','XGBoost'),('C','RF')]:bees(axes[k],arrays(cfg,n+'_full_shap'));axes[k].set_title(DISPLAY.get(n,n),fontsize=10)
    for n,marker in [('XGBoost','o'),('RF','s')]:
        t=red[red.Model==n].sort_values('Feature_Count');axes['D'].plot(t.Feature_Count,t.AUROC,color=COLORS[n],marker=marker,ms=3,lw=1.4,label=DISPLAY.get(n,n))
    axes['D'].set(ylabel='AUROC');axes['D'].legend(loc='lower right')
    e=[]
    for k in range(4,29):
        p=oo[f'k_{k}'];fpr,tpr,cut=roc_curve(oo.D2,p);valid=np.isfinite(cut);threshold=cut[valid][np.argmax((tpr-fpr)[valid])]
        tn,fp,fn,tp=confusion_matrix(oo.D2,p>=threshold,labels=[0,1]).ravel()
        e.append({'Feature_Count':k,'Sensitivity':tp/(tp+fn),'Specificity':tn/(tn+fp),'F1':2*tp/(2*tp+fp+fn),'descriptive_threshold':threshold})
    e=pd.DataFrame(e).merge(red[red.Model=='XGBoost'][['Feature_Count','AUROC']],on='Feature_Count');save_table(cfg,'Figure2E_descriptive',e)
    for m,c,mark in [('AUROC','#2ca02c','o'),('Specificity','#1f77b4','s'),('Sensitivity','#d62728','^'),('F1','#ff7f0e','D')]:axes['E'].plot(e.Feature_Count,e[m],color=c,marker=mark,ms=3,label=m)
    axes['E'].set(ylabel='Performance metrics',ylim=(0,1.015),title='XGBoost');axes['E'].legend(loc='lower right',ncol=2)
    for k in 'DE':axes[k].axvline(9,color='black',ls='--',lw=1.1);axes[k].set(xlabel='Number of features',xlim=(3.5,28.5));axes[k].set_xticks([4,8,12,16,20,24,28])
    for i,c in zip(range(1,6),['#8cbde6','#ffbb88','#f69b9f','#c3abe1','#bfaaa6']):
        t=final[final.fold==i];x,y,_=roc_curve(t.D2,t.XGBoost);axes['F'].plot(x,y,color=c,lw=.8,label=f'Fold {i}')
    from .models import metrics
    x,y,_=roc_curve(final.D2,final.XGBoost);a=metrics(final.D2,final.XGBoost)['AUROC'];axes['F'].plot(x,y,color=COLORS['XGBoost'],lw=2,label=f'XGBoost (AUROC = {a:.3f})')
    axes['F'].plot([0,1],[0,1],'--',color='.65');axes['F'].set(xlim=(0,1),ylim=(0,1.015),xlabel='False Positive Rate',ylabel='True Positive Rate');axes['F'].legend(loc='lower right',fontsize=8)
    letters(fig,[('A',.006,.995),('B',.341,.995),('C',.677,.995),('D',.006,.49),('E',.341,.49),('F',.677,.49)]);save(cfg,fig,'Figure2')
    fig,ax=plt.subplots(figsize=(6,4.8))
    for _,r in met[met.Model!='Baseline'].sort_values('AUPRC',ascending=False).iterrows():
        p,rr,_=precision_recall_curve(dev.D2,dev[r.Model]);ax.plot(rr,p,color=COLORS[r.Model],label=f'{DISPLAY.get(r.Model,r.Model)} (AP = {r.AUPRC:.3f})')
    ax.axhline(dev.D2.mean(),ls='--',color='.5',label=f'D2 prevalence = {dev.D2.mean():.3f}');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Recall',ylabel='Precision');ax.legend(fontsize=8);fig.tight_layout();save(cfg,fig,'FigureS2')

def calibration_plot(cfg,ax,stage):
    c=table(cfg,'Calibration');c=c[(c.Cohort=='external')&(c.Stage==stage)].iloc[0]
    b=table(cfg,'external_raw_calibration_bins' if stage=='Raw' else 'external_crossfit_calibration_bins')
    grid=np.linspace(1e-8,1-1e-8,501) if stage=='Raw' else np.linspace(.000001,.999999,1001)
    curve=expit(c.Calibration_intercept+c.Calibration_slope*logit(grid))
    ax.plot(grid,curve,color=COLORS['XGBoost'],lw=1.7,label='XGBoost');ax.scatter(b.Mean_predicted,b.Observed,s=28,color=COLORS['XGBoost'],zorder=3)
    ax.plot([0,1],[0,1],'--',color='.65',label='Ideal');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed D2 proportion');ax.legend(loc='lower right')

def figure3(cfg):
    style();p=load_private(cfg,'external_predictions');perf=table(cfg,'Performance_CI');net=table(cfg,'DCA_raw_full_range')
    fig=plt.figure(figsize=(11,9.6));axes={k:fig.add_axes(pos) for k,pos in {'A':[.08,.57,.39,.37],'B':[.58,.57,.39,.37],'C':[.08,.08,.39,.37],'D':[.58,.08,.39,.37]}.items()}
    for k,m in [('A','AUROC'),('B','AUPRC')]:
        ax=axes[k];r=perf[(perf.Cohort=='external')&(perf.Model=='XGBoost')&(perf.Metric==m)].iloc[0]
        if k=='A':x,y,_=roc_curve(p.D2,p.XGBoost)
        else:y,x,_=precision_recall_curve(p.D2,p.XGBoost)
        ax.plot(x,y,color=COLORS['XGBoost'],lw=1.6,label=f'XGBoost\n{m} = {r.Estimate:.3f}\n95% CI {r.CI_lower:.3f}–{r.CI_upper:.3f}')
        if k=='A':ax.plot([0,1],[0,1],'--',color='.65');ax.set(xlabel='False Positive Rate',ylabel='True Positive Rate')
        else:ax.axhline(p.D2.mean(),ls='--',color='.65',label=f'D2 prevalence = {p.D2.mean():.3f}');ax.set(xlabel='Recall',ylabel='Precision')
        ax.set(xlim=(0,1),ylim=(0,1));ax.legend(loc='lower right' if k=='A' else 'upper right')
    ax=axes['C']
    for field,label,c,lw in [('XGBoost','XGBoost',COLORS['XGBoost'],2),('Treat_all','Treat all','.55',1.3),('Treat_none','Treat none','black',1.25)]:ax.plot(net.threshold,net[field],color=c,lw=lw,label=label)
    ax.set(xlim=(0,.12),ylim=(-.02,.075),xlabel='Threshold probability',ylabel='Net benefit');ax.set_xticks([0,.03,.06,.09,.12]);ax.set_yticks([-.02,0,.02,.04,.06]);ax.legend(loc='upper right')
    calibration_plot(cfg,axes['D'],'Cross-fitted local recalibration')
    letters(fig,[('A',.012,.977),('B',.512,.977),('C',.012,.492),('D',.512,.492)]);save(cfg,fig,'Figure3')
    fig,ax=plt.subplots(figsize=(6.2,5.7));calibration_plot(cfg,ax,'Raw');ax.set_aspect('equal');ax.set(xlabel='Original predicted probability',ylabel='Observed probability');fig.tight_layout();save(cfg,fig,'FigureS3')

def figure4(cfg):
    style();z=arrays(cfg,'final_shap');features=FROZEN['final'];raw=load_private(cfg,'development');rank=table(cfg,'Final_SHAP_ranking');order=[features.index(f) for f in rank.feature]
    fig=plt.figure(figsize=(12,14));ax=fig.add_axes([.115,.715,.34,.255]);ax.barh(np.arange(9),rank.mean_abs_SHAP,color='#5797b4',height=.65);ax.set_yticks(np.arange(9),rank.feature);ax.invert_yaxis();ax.set_xlabel('Mean |SHAP value|');ax.set_xlim(0,rank.mean_abs_SHAP.max()*1.16)
    for i,v in enumerate(rank.mean_abs_SHAP):ax.text(v+.012,i,f'{v:.3f}',va='center')
    bees(fig.add_axes([.61,.715,.31,.255]),z,max_display=9,order=np.array(order))
    units={'age':'years','eGFRcr_2021':'mL/min/1.73 m²','HGB':'g/L','PLT':'10⁹/L','BUN':'mmol/L','ALB':'g/L','ALP':'U/L','GLU':'mmol/L'}
    gs=fig.add_gridspec(3,3,left=.08,right=.98,bottom=.055,top=.62,hspace=.48,wspace=.38)
    for j,f in enumerate(features):
        ax=fig.add_subplot(gs[j//3,j%3]);valid=raw[f].notna();x=raw.loc[valid,f].to_numpy();y=z['values'][valid,j];px=x.copy()
        if f=='sex':px=px+np.random.default_rng(42).uniform(-.075,.075,len(px))
        ax.axhline(0,color='.55',ls='--',lw=.8);ax.scatter(px,y,color='#2389c9',s=8,alpha=.43,edgecolors='none')
        if f!='sex':
            curve=lowess(y,x,frac=.35,it=3,return_sorted=True);lo,hi=np.quantile(x,[.05,.95]);curve=curve[(curve[:,0]>=lo)&(curve[:,0]<=hi)];ax.plot(curve[:,0],curve[:,1],color='#b04c46',lw=1.5)
        else:ax.set_xticks([0,1]);ax.set_xlim(-.22,1.22)
        label={'age':'Age','eGFRcr_2021':'eGFRcr'}.get(f,f);ax.set(xlabel=label+(' ('+units[f]+')' if f in units else ''),ylabel='SHAP value');ax.margins(y=.12)
    letters(fig,[('A',.008,.995),('B',.505,.995),('C',.008,.662)]);save(cfg,fig,'Figure4')

def figure5(cfg):
    style();t=table(cfg,'Targeted_testing');fig=plt.figure(figsize=(11.4,9.4));spec={'A':('D2_recall','D2 cases detected (%)',100),'B':('Screening_yield','D2 yield among tested (%)',100),'C':('Tests_per_D2_detected','Cys-C tests per D2 detected',1),'D':('Enrichment_vs_random','Enrichment vs random',1)}
    for k,pos in {'A':[.085,.58,.38,.36],'B':[.59,.58,.37,.36],'C':[.085,.09,.38,.36],'D':[.59,.09,.37,.36]}.items():
        ax=fig.add_axes(pos);m,label,scale=spec[k]
        for co,marker in [('development','o'),('external','s')]:
            d=t[t.Cohort==co].sort_values('Budget_percent');x=d.Budget_percent
            ax.fill_between(x,d[m+'_CI_lower']*scale,d[m+'_CI_upper']*scale,color=CC[co],alpha=.13,linewidth=0);ax.plot(x,d[m]*scale,color=CC[co],marker=marker,lw=1.8,label=COHORTS[co],clip_on=False)
        if k=='A':ax.plot([0,50],[0,50],'--',color='.5',label='Random testing');ax.set_ylim(0,100)
        elif k=='B':
            for co in COHORTS:ax.axhline(100*t[t.Cohort==co].Cohort_prevalence.iloc[0],ls='--',color=CC[co],alpha=.6,label='Development prevalence' if co=='development' else 'External prevalence')
            ax.set_ylim(0,80)
        else:
            ax.set_ylim(0,np.ceil(t[m+'_CI_upper'].max()))
            if k=='D':ax.axhline(1,color='.5',ls='--',label='Random testing')
        ax.set(xlim=(0,50),xlabel='Proportion tested (%)',ylabel=label);ax.set_xticks([10,20,30,40,50]);ax.legend(loc='lower right' if k=='A' else 'upper right')
    letters(fig,[('A',.015,.985),('B',.515,.985),('C',.015,.495),('D',.515,.495)]);save(cfg,fig,'Figure5')

def figure6(cfg):
    style();df=table(cfg,'Four_group_profile');fig=plt.figure(figsize=(13.6,10.4));ticks=[s.replace(' / ','\n') for s in GROUPS]
    for k,pos in {'A':[.08,.59,.37,.32],'B':[.58,.59,.38,.32],'C':[.08,.14,.37,.32]}.items():
        ax=fig.add_axes(pos)
        for ci,co in enumerate(COHORTS):
            d=df[df.Cohort==co].sort_values('Group_index');xx=np.arange(4)+(ci-.5)*.27
            if k=='A':
                y=d.Cohort_fraction.to_numpy()*100;ax.bar(xx,y,width=.25,color=CC[co],label=COHORTS[co])
                for x,v,n in zip(xx,y,d.N):ax.text(x,v+1.5,f'{n}\n({v:.1f}%)',ha='center',fontsize=9)
            else:
                m='D1' if k=='B' else 'D3';y=d[m+'_prevalence'].to_numpy()*100;lo=d[m+'_CI_lower'].to_numpy()*100;hi=d[m+'_CI_upper'].to_numpy()*100
                ax.errorbar(xx,y,yerr=[y-lo,hi-y],fmt='o' if ci==0 else 's',color=CC[co],ms=6,capsize=3,label=COHORTS[co])
                if k=='C':
                    for x,v,n in zip(xx,hi,d.D3_denominator):ax.annotate(f'n={n}',(x,v),xytext=(-3,6) if ci==0 else (3,10),textcoords='offset points',ha='right' if ci==0 else 'left',va='bottom',fontsize=8)
        ax.set_xticks(np.arange(4),ticks);ax.set(xlim=(-.55,3.55),ylim=(0,100),ylabel={'A':'Participants (%)','B':'D1 prevalence (%)','C':'D3 prevalence (%)\namong eGFRcr ≥60'}[k]);ax.legend(loc='upper right' if k=='A' else 'lower right',fontsize=9)
    fields=[('D1_prevalence','D1 (%)',100),('D3_prevalence','D3 (%)¹',100),('eGFRcys_lt60_prevalence','eGFRcys <60 (%)',100),('eGFRcys_2012_median','eGFRcys²',1),('eGFR_difference_median','eGFR difference²',1),('CysC_mg_L_median','Cys-C (mg/L)',1)]
    ordered=pd.concat([df[df.Cohort==co].sort_values('Group_index') for co in COHORTS]);mat=np.array([ordered[f].to_numpy()*s for f,_,s in fields]);std=mat.std(axis=1,keepdims=True);z=np.divide(mat-mat.mean(axis=1,keepdims=True),std,out=np.zeros_like(mat),where=std!=0)
    for ci,co in enumerate(COHORTS):
        ax=fig.add_axes([.66+ci*.30*.52,.185,.30*.46,.265]);a=z[:,ci*4:ci*4+4];v=mat[:,ci*4:ci*4+4];im=ax.imshow(a,cmap='RdBu_r',vmin=-2.5,vmax=2.5,aspect='auto');ax.set_xticks(range(4),ticks,fontsize=8);ax.set_yticks(range(6),[r[1] for r in fields] if ci==0 else ['']*6,fontsize=9);ax.tick_params(length=0);ax.set_title(COHORTS[co],fontsize=10,pad=9)
        for i in range(6):
            for j in range(4):ax.text(j,i,f'{v[i,j]:.2f}' if i==5 else f'{v[i,j]:.1f}',ha='center',va='center',fontsize=9,color='white' if abs(a[i,j])>1.7 else 'black')
        ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True);ax.grid(which='minor',color='white',lw=1);ax.tick_params(which='minor',length=0)
        for s in ax.spines.values():s.set_visible(False)
    cb=fig.colorbar(im,cax=fig.add_axes([.726,.093,.174,.012]),orientation='horizontal',ticks=[-2,0,2]);cb.set_label('Within-row z score (shared across cohorts)',fontsize=8);cb.ax.tick_params(labelsize=8)
    fig.text(.66,.047,'¹ Denominator: eGFRcr ≥60.  ² mL/min/1.73 m²; medians.',fontsize=7)
    for x,y,title in [(.08,.936,'Four-group composition'),(.58,.936,'D1: absolute eGFR discordance'),(.08,.486,'D3: creatinine-masked low eGFR'),(.58,.486,'Renal phenotype profiles')]:fig.text(x,y,title,fontsize=12,ha='left',va='bottom')
    letters(fig,[('A',.015,.975),('B',.515,.975),('C',.015,.515),('D',.515,.515)]);save(cfg,fig,'Figure6')

def run(cfg):
    require(cfg,'profiles');require(cfg,'shap')
    for func in [figure2,figure3,figure4,figure5,figure6]:func(cfg);print('Rendered',func.__name__,flush=True)
    style();c=table(cfg,'S1_correlation_matrix').set_index('feature');fig,ax=plt.subplots(figsize=(12,11));sns.heatmap(c,vmin=-1,vmax=1,center=0,cmap='RdBu_r',square=True,ax=ax,cbar_kws={'shrink':.65,'label':'Spearman rho'});ax.set_title(f'Pre-redundancy correlation in the development cohort (n = {len(load_private(cfg,"development"))})',pad=15);ax.tick_params(axis='x',rotation=90);ax.tick_params(axis='y',rotation=0);fig.tight_layout();save(cfg,fig,'FigureS1')
    checkpoint(cfg,'figures')
