"""Narrative-led scientific figures from frozen exported values; no fitting or resampling."""

import publication_style as ps
from pathlib import Path
import argparse,hashlib,json,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator,FixedLocator
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageDraw
if Path(__file__).with_name('explanation_figures.py').is_file():
 from explanation_figures import response_figure,term_importance_figure
else:
 from reporting.manuscript_workspace.explanation_figures import response_figure,term_importance_figure
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'03_FIGURES/narrative_revision'
QA=OUT/'qa'
MM=1/25.4
BLUE,ORANGE,GREEN=(ps.INPUT_COLORS[key] for key in ('preop_only','all_waveforms','preop_and_all_waveforms'))
OUTCOMES=[('any_aki','AKI'),('icu_admission','ICU admission')]
FAMILIES=[('Logistic regression','LR'),('XGBoost','XGBoost'),('EBM','EBM')]
SOURCES={
 'metrics':ROOT/'04_DATA/results/main_shared/tables/metrics_summary.csv',
 'importance':ROOT/'02_TABLES/inputs/Table_S6_EBM_fold_importance.csv',
 'labels':ROOT/'02_TABLES/inputs/feature_labels.json',
 'responses':ROOT/'03_FIGURES/paper/figure_4_ebm_response_functions.csv',
 'support':ROOT/'03_FIGURES/paper/figure_4_training_support.csv',
 'calibration':ROOT/'03_FIGURES/paper/S4-whole-case-calibration.plotted.csv',
 'extended_responses':ROOT/'03_FIGURES/paper/S6-ebm-nonlinearity-missingness.plotted.csv'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def path_label(path):return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
BEFORE={}
ps.apply_mpl_style()
MANIFEST=[];CHECKS={}
def canvas(height):return plt.figure(figsize=(ps.WIDTH_MM*MM,height*MM))
def clean(ax,grid=False):
 ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True)
 if grid:ax.grid(axis='y',color=ps.color('grid'),lw=ps.GRID_WIDTH)
def title(ax,letter,label):ax.set_title(r'$\bf{'+letter+'}$  '+label,loc='left',pad=6)
def save(fig,stem,label,heading,caption,plotted,used,*,png_only=False,**details):
 fig.canvas.draw();w,h=fig.get_size_inches()*25.4
 assert abs(w-ps.WIDTH_MM)<1e-8 and h<=234
 visible=list(fig.texts);axes=[]
 def descend(ax):
  axes.append(ax)
  for child in ax.child_axes:descend(child)
 for ax in fig.axes:descend(ax)
 for ax in axes:
  visible+=ax.texts+[ax.title,ax._left_title,ax._right_title]
  if ax.axison:
   visible +=[ax.xaxis.label,ax.yaxis.label]
   for ticks,texts,lim in [(ax.get_xticks(),ax.get_xticklabels(),ax.get_xlim()),(ax.get_yticks(),ax.get_yticklabels(),ax.get_ylim())]:
    visible +=[t for v,t in zip(ticks,texts) if min(lim)-1e-9<=v<=max(lim)+1e-9]
 for legend in fig.legends:visible+=legend.get_texts()
 renderer=fig.canvas.get_renderer();outside=[]
 visible=[t for t in visible if t.get_visible() and t.get_text()]
 for t in visible:
  box=t.get_window_extent(renderer)
  if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:outside.append(t.get_text())
 assert not outside,(stem,outside)
 fonts=sorted(set(float(t.get_fontsize()) for t in visible));assert min(fonts)>=8 and max(fonts)<=10,(stem,fonts)
 formats=ps.export_figure(fig, OUT/stem, png_only=png_only)
 if not png_only:
  assert '<image' not in (OUT/(stem+'.svg')).read_text()
 plt.close(fig)
 plotted.to_csv(OUT/(stem+'.plotted.csv'),index=False,float_format='%.17g')
 (OUT/(stem+'.caption.txt')).write_text(caption.strip()+'\n')
 entry={'style_profile_sha256':ps.STYLE_SHA256,'stem':stem,'label':label,'title':heading,'folder':path_label(OUT),
  'width_mm':float(w),'height_mm':float(h),'font_family':ps.FONT_FAMILY,'font_sizes_pt':fonts,
  'caption_file':stem+'.caption.txt','plotted_data_file':stem+'.plotted.csv','plotted_rows':len(plotted),
  'formats':formats,
  'drawn_text_bounds_verified':True,'drawn_text_count':len(visible),
  'sources':[{'path':path_label(SOURCES[k]),'sha256':BEFORE[path_label(SOURCES[k])]} for k in used],**details}
 if not png_only:
  entry['svg_has_embedded_raster']=False
 MANIFEST.append(entry);(OUT/(stem+'.metadata.json')).write_text(json.dumps(entry,indent=2)+'\n')
 print(stem,round(h,1),'mm',flush=True)


def incremental():
 source=pd.read_csv(SOURCES['metrics']);d=source[source.branch.eq('non_windowed')&source.feature_set.eq('preop_and_all_waveforms')].copy()
 assert len(d)==6 and d.reference_feature_set.eq('preop_only').all()
 assert (d.model_name==d.reference_model_name).all()
 rows=[]
 for outcome,_ in OUTCOMES:
  for family,_ in FAMILIES:
   r=d[d.outcome.eq(outcome)&d.model_name.eq(family)].iloc[0]
   for metric,name in [('auprc','AP'),('auroc','AUROC')]:
    rows.append({'outcome':outcome,'branch':'non_windowed','candidate_model':family,'candidate_inputs':'preop_and_all_waveforms','reference_model':family,'reference_inputs':'preop_only','metric':metric,'metric_label':name,'delta':r['delta_'+metric],'lower':r['delta_'+metric+'_lower'],'upper':r['delta_'+metric+'_upper'],'unit':'0–1 scale','bootstrap_draw_hash':r.bootstrap_draw_hash})
 data=pd.DataFrame(rows)
 fig=canvas(144);grid=fig.add_gridspec(2,2,left=.115,right=.985,bottom=.16,top=.93,hspace=.62,wspace=.37)
 for ri,(outcome,olabel) in enumerate(OUTCOMES):
  for ci,(metric,name) in enumerate([('auprc','AP'),('auroc','AUROC')]):
   sub=grid[ri,ci].subgridspec(1,2,width_ratios=[1,.93],wspace=.07)
   ax=fig.add_subplot(sub[0]);nums=fig.add_subplot(sub[1],sharey=ax);nums.axis('off')
   title(ax,chr(97+ri*2+ci),olabel)
   nums.text(1,1.05,'Difference (95% CI)',transform=nums.transAxes,ha='right',va='bottom',fontsize=ps.STYLE['figure']['body_pt'])
   for i,(family,_) in enumerate(FAMILIES):
    r=data[data.outcome.eq(outcome)&data.metric.eq(metric)&data.candidate_model.eq(family)].iloc[0]
    v,l,u=100*np.array([r.delta,r.lower,r.upper]);y=2-i
    ax.errorbar(v,y,xerr=[[v-l],[u-v]],fmt='o',color=GREEN,capsize=2.2,lw=1,ms=4.3)
    nums.text(1,y,f'{v:+.1f} ({l:+.1f}, {u:+.1f})'.replace('-','−'),transform=nums.get_yaxis_transform(),ha='right',va='center',fontsize=ps.STYLE['figure']['body_pt'])
   ax.axvline(0,color=ps.color('reference'),ls='--',lw=ps.REFERENCE_WIDTH)
   limits=(-14,11) if metric=='auprc' else (-6.5,5)
   ax.set(xlim=limits,ylim=(-.55,2.55),yticks=[2,1,0],yticklabels=[f[1] for f in FAMILIES]);ax.tick_params(axis='y',length=0)
   ax.xaxis.set_major_locator(MaxNLocator(3));ax.spines['left'].set_visible(False);clean(ax);ax.grid(axis='x',color=ps.color('grid'),lw=ps.GRID_WIDTH)
 fig.text(.30,.985,'AP',fontsize=ps.STYLE['figure']['title_pt'],weight='bold',ha='center',va='top')
 fig.text(.80,.985,'AUROC',fontsize=ps.STYLE['figure']['title_pt'],weight='bold',ha='center',va='top')
 fig.supxlabel('Combined − clinical (percentage points)',fontsize=ps.STYLE['figure']['body_pt'],y=.035)
 caption='Whole-case changes in average precision (AP) and area under the receiver operating characteristic curve (AUROC): combined minus same-family clinical-only models. Points and whiskers show estimates and 95% paired patient-cluster bootstrap confidence intervals in percentage points. Dashed lines mark zero.'
 save(fig,'Figure2_incremental_AP_AUROC','Figure 2','Incremental discrimination',caption,data,['metrics'],comparison_count=6,metric_count=2,primary_metric='AP',scientific_values_recomputed=False)
 CHECKS['incremental']={'contrasts':6,'metric_entries':12,'all_same_family_references':True,'all_saved_CIs_retained':True}


def calibration():
 from build_calibration_support import load_support
 source=pd.read_csv(SOURCES['calibration']);d=source[source.feature_set.isin(['preop_only','preop_and_all_waveforms'])].copy();assert len(d)==120
 support,support_checks,support_hashes=load_support(ROOT)
 used=['calibration']
 for i,(path,digest) in enumerate(support_hashes.items()):
  key=f'calibration_support_{i}';SOURCES[key]=ROOT/path;BEFORE[path]=digest;used.append(key)
 fig=canvas(184);grid=fig.add_gridspec(2,3,left=.10,right=.98,bottom=.16,top=.94,hspace=.49,wspace=.31)
 inputs=[(key,ps.INPUT_COLORS[key],ps.INPUT_STYLES[key],ps.INPUT_MARKERS[key],ps.INPUT_LABELS[key]) for key in ('preop_only','preop_and_all_waveforms')]
 for ri,(outcome,olabel) in enumerate(OUTCOMES):
  support_max=support[support.outcome.eq(outcome)]['count'].max()
  for ci,(family,flabel) in enumerate([('logistic_regression','LR'),('xgboost','XGBoost'),('ebm','EBM')]):
   sub=grid[ri,ci].subgridspec(2,1,height_ratios=[3.6,.85],hspace=.06)
   ax=fig.add_subplot(sub[0]);hx=fig.add_subplot(sub[1],sharex=ax)
   title(ax,chr(97+ri*3+ci),f'{olabel} · {flabel}')
   for inp,color,style,marker,_ in inputs:
    z=d[d.outcome.eq(outcome)&d.family.eq(family)&d.feature_set.eq(inp)].sort_values('bin');assert len(z)==10
    ax.plot(z.mean_risk,z.observed_rate,color=color,ls=style,marker=marker,ms=3,lw=ps.CURVE_WIDTH)
    h=support[support.outcome.eq(outcome)&support.family.eq(family)&support.feature_set.eq(inp)].sort_values('bin')
    edges=np.r_[h.bin_lower.to_numpy(),h.bin_upper.iloc[-1]]
    hx.stairs(h['count'],edges,color=color,ls=style,lw=ps.CURVE_WIDTH)
   ax.plot([0,1],[0,1],color=ps.color('reference'),ls='--',lw=ps.REFERENCE_WIDTH)
   ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1]);ax.set_aspect('equal');ax.set_anchor('S');clean(ax,True)
   ax.tick_params(axis='x',labelbottom=False)
   hx.set(xlim=(0,1),ylim=(0,support_max*1.08),xticks=[0,.5,1],xlabel='Predicted risk')
   hx.yaxis.set_major_locator(MaxNLocator(2,integer=True));clean(hx)
   if ri==0:
    detail=ax.inset_axes([.49,.43,.48,.43])
    for inp,color,style,marker,_ in inputs:
     z=d[d.outcome.eq(outcome)&d.family.eq(family)&d.feature_set.eq(inp)].sort_values('bin')
     detail.plot(z.mean_risk,z.observed_rate,color=color,ls=style,marker=marker,ms=2,lw=0.75 * ps.CURVE_WIDTH)
    detail.plot([0,.35],[0,.35],color=ps.color('reference'),ls='--',lw=ps.REFERENCE_WIDTH)
    detail.set(xlim=(0,.35),ylim=(0,.35),xticks=[0,.3],yticks=[0,.3]);detail.tick_params(labelsize=8,length=1.5,pad=1)
    detail.set_title('0–0.35',fontsize=ps.STYLE['figure']['body_pt'],pad=3);clean(detail)
    for spine in detail.spines.values():spine.set_linewidth(.5)
    ax.add_patch(Rectangle((0,0),.35,.35,fill=False,edgecolor=ps.color('reference'),lw=.5,ls=':'))
   if ci==0:ax.set_ylabel('Observed risk');hx.set_ylabel('Held-out n')
 handles=[Line2D([0],[0],color=c,ls=s,marker=m,lw=ps.CURVE_WIDTH,label=l) for _,c,s,m,l in inputs]
 handles.append(Line2D([0],[0],color=ps.color('reference'),ls='--',lw=ps.REFERENCE_WIDTH,label='Perfect calibration'))
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False,handlelength=2.6,columnspacing=1.5)
 stem='Figure3_calibration_clinical_combined';support_file=stem+'.prediction_support.csv'
 support.to_csv(OUT/support_file,index=False,float_format='%.17g')
 caption='Whole-case clinical-only and combined calibration from pooled outer-test predictions. Points show observed frequency against mean predicted risk in ten quantile groups; dashed diagonals show perfect calibration. Insets show AKI risks from 0 to 0.35. Lower panels show prediction counts in 0.025-wide risk bins; count scales are shared within outcomes. Diagnostics: Table S8.'
 save(fig,stem,'Figure 3','Calibration',caption,d,used,configurations=12,quantile_groups=10,learned_calibration_recomputed=False,support_data_file=support_file,support_kind='held-out calibrated probability histograms',histogram_bins=40,histogram_bin_width=.025)
 CHECKS['calibration']={'rows':120,'configurations':12,'points_per_model':10,'saved_bin_counts_retained':True,'outer_test_data_only':True,'support_rows':len(support),'support_checks':support_checks,'support_source_sha256':support_hashes}


def importance(revision=False):
 source=pd.read_csv(SOURCES['importance']);labels=json.loads(SOURCES['labels'].read_text())
 d=source[source.branch.eq('non_windowed')&source.feature_set.eq('preop_and_all_waveforms')].copy();d['fold_number']=d.fold.str.removeprefix('outer_').astype(int)+1
 custom={'optype_Transplantation':'Transplantation','optype_Colorectal':'Colorectal procedure','optype_Major resection':'Major resection','department_Thoracic surgery':'Thoracic surgery department','department_General surgery':'General surgery department','asa_3.0':'ASA III','asa_1.0':'ASA I','SNUADC_ECG_II_CO_HistogramAMI_even_2_5':'ECG · Auto mutual information (even bins)','Primus_AWP_SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1':'AWP · Rescaled-range fluctuation','Primus_CO2_SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1':'CO₂ · Detrended fluctuation (50%)','Primus_AWP_SB_BinaryStats_diff_longstretch0':'AWP · Longest stretch of successive differences','SNUADC_PLETH_DN_Mean':'PLETH · Processed mean'}
 selected=[];display=[]
 for outcome,_ in OUTCOMES:
  g=d[d.outcome.eq(outcome)]
  means=g.groupby('feature',sort=False).native_importance.mean().sort_values(ascending=False,kind='stable').head(10)
  for j,(feature,mean) in enumerate(means.items()):
   z=g[g.feature.eq(feature)].sort_values('fold_number');assert len(z)==5 and set(z.fold_number)==set(range(1,6))
   group=z.group.iloc[0]
   label=custom.get(feature,labels.get(feature,feature)).replace('EtCO₂','CO₂').replace(' — ',' · ')
   if group=='clinical':label='Clinical · '+label
   display_label=label.removeprefix('Clinical · ').replace('AWP · Longest stretch of successive differences', 'AWP · Decreasing-run statistic')
   if revision:
    display_label=(display_label
                   .replace('ASA ', 'ASA-PS ')
                   .replace('Auto mutual information (even bins)', 'Lag-2 auto mutual information')
                   .replace('Detrended fluctuation (50%)', 'DFA crossover fraction')
                   .replace('CO₂ · ', 'Capnography · ')
                   .replace('AWP · ', 'Airway pressure · ')
                   .replace('PLETH · Processed mean', 'PPG · Filtered signal mean'))
   z=z.assign(mean_native_importance=mean,display_rank=j+1,display_label=display_label if revision else label);selected.append(z)
   display.append(z.assign(display_label=display_label))
 fig=term_importance_figure(pd.concat(display,ignore_index=True), [('any_aki','AKI',.72),('icu_admission','ICU admission',.35)])
 data=pd.concat(selected,ignore_index=True);assert len(data)==100
 caption='Ten leading terms in whole-case combined explainable boosting machine (EBM) models, ranked by mean importance across five outer folds. Gray bars show means; symbols show folds. Importance is the training-weighted mean absolute raw term score (log-odds). Channel prefixes identify waveform terms. Horizontal scales differ by outcome. AWP, airway pressure; PLETH, plethysmography. Waveform definitions: Table S9.'
 if revision:
  caption=('Importance represents the mean absolute contribution to model log-odds, weighted by the training distribution. '
           'Bars show means across five folds; symbols show individual folds. AKI, acute kidney injury; '
           'ICU, intensive care unit; EBM, explainable boosting machine; ASA-PS, American Society of Anesthesiologists physical status; '
           'DFA, detrended fluctuation analysis; ECG, electrocardiography; PPG, photoplethysmography.')
 save(fig,'Figure4_EBM_global_importance','Figure 4','EBM term importance',caption,data,['importance','labels'],png_only=revision,terms_per_outcome=10,folds=5,selection='Descending across-fold mean native importance; top ten independently per outcome',group_importance_sums_used=False)
 CHECKS['importance']={'rows':100,'terms_per_outcome':10,'five_folds_per_term':True,'native_importance_only':True,'all_nonnegative':bool(data.native_importance.ge(0).all()),'importance_definition':'training-distribution weighted mean absolute native term contribution'}


def responses(replacement=None):
 source=pd.read_csv(SOURCES['responses']);support=pd.read_csv(SOURCES['support'])
 selected_source_panels=['A','C','D'] if replacement is None else ['A','C']
 data=source[source.panel.isin(selected_source_panels)].copy()
 hist=support[support.panel.isin(selected_source_panels)].copy()
 last_panel=('D','ICU admission · CO₂','Time-reversibility statistic',FixedLocator([-.0075,-.003,0]))
 used=['responses','support']
 details={}
 if replacement is not None:
  data=pd.concat([data,pd.DataFrame(replacement['curve']).assign(panel='D')[data.columns]],ignore_index=True)
  hist=pd.concat([hist,pd.DataFrame(replacement['training_support']).assign(panel='D')[hist.columns]],ignore_index=True)
  last_panel=('D','ICU admission · eGFR','eGFR (mL/min/1.73 m²)',FixedLocator([65,80,95,110]))
  used.append('replacement')
  details['replacement_term']=replacement['provenance']
 fig=response_figure(data,hist,'panel',[
  ('A','AKI · Hemoglobin','Hemoglobin (g/dL)',FixedLocator([10,13,16])),
  ('C','ICU admission · Age','Age (years)',FixedLocator([45,60,75])),last_panel,
 ])
 stem='Figure5_EBM_selected_functions';hist.to_csv(OUT/(stem+'.training_support.csv'),index=False,float_format='%.17g')
 caption='Whole-case combined EBM functions: (a) hemoglobin for AKI; (b) age and (c) CO₂ time reversibility for ICU admission. Lines show fold-specific raw log-odds relative to the labeled reference, before recalibration; histograms show finite values in the first training fold. Hemoglobin spans median fold-specific 1st–99th training percentiles; age and CO₂ span 5th–95th percentiles.'
 if replacement is not None:
  caption=('Panels show (a) hemoglobin for AKI and (b) age and (c) preoperative estimated glomerular '
           'filtration rate (eGFR) for ICU admission. '
           'Lines show fold-specific log-odds relative to the labeled reference before recalibration. '
           'Histograms show training distributions from the first fold.')
 save(fig,stem,'Figure 5','EBM response functions',caption,data,used,png_only=replacement is not None,selected_source_panels=selected_source_panels,support_data_file=stem+'.training_support.csv',folds=5,recomputed_term_scores=False,**details)
 CHECKS['responses']={'response_rows':len(data),'training_support_rows':len(hist),'selected_source_panels':selected_source_panels,'folds_per_term':5,'scores_recentered_or_refitted':False}
 if replacement is not None:
  CHECKS['responses']['replacement_term']=replacement['provenance']['feature']


def extended_responses(replacement=None):
 original=pd.read_csv(SOURCES['responses']);support=pd.read_csv(SOURCES['support']);extended=pd.read_csv(SOURCES['extended_responses'])
 awp=original[original.panel.eq('B')].copy();awp['display_panel']='a'
 awph=support[support.panel.eq('B')].copy();awph['display_panel']='a'
 frames=[awp];histograms=[awph]
 source_panels=[('b','AKI · Observed WBC response','any_aki','preop_wbc')]
 if replacement is None:
  source_panels.append(('c','ICU · Processed pleth mean response','icu_admission','SNUADC_PLETH_DN_Mean'))
 for key,name,outcome,feature in source_panels:
  r=extended[extended.panel.eq(name)].copy();h=extended[extended.panel.eq(name+' training histogram')].copy()
  r['outcome']=outcome;r['feature_set']='preop_and_all_waveforms';r['feature']=feature;r['fold']=r['fold']-1;r['reference']=r.reference_value;r['raw_contribution_minus_reference']=r.raw_contribution;r['display_panel']=key
  h['outcome']=outcome;h['feature']=feature;h['fold']=h['fold']-1;h['display_panel']=key
  frames.append(r);histograms.append(h)
 if replacement is not None:
  frames.append(pd.DataFrame(replacement['curve']).assign(display_panel='c'))
  histograms.append(pd.DataFrame(replacement['training_support']).assign(display_panel='c'))
 data=pd.concat(frames,ignore_index=True)[['display_panel','outcome','feature_set','feature','fold','reference','value','raw_contribution_minus_reference']]
 hist=pd.concat(histograms,ignore_index=True)[['display_panel','outcome','feature','fold','bin_lower','bin_upper','count']]
 last_panel=('c','ICU admission · PLETH','Filtered waveform mean',FixedLocator([-.0015,0,.0015]))
 if replacement is not None:
  last_panel=('c','ICU · Capnography','Mean capnogram CO₂\n(nominal mmHg)',MaxNLocator(4,integer=True))
 fig=response_figure(data,hist,'display_panel',[
  ('a','AKI · Airway pressure','Rescaled-range fluctuation',FixedLocator([.2,.5,.8])),
  ('b','AKI · WBC','WBC (10³/µL)',FixedLocator([4,8,12,16])),last_panel,
 ])
 stem='S10_EBM_extended_functions' if replacement is not None else 'S9_EBM_extended_functions'
 hist.to_csv(OUT/(stem+'.training_support.csv'),index=False,float_format='%.17g')
 caption='Additional whole-case combined EBM functions: (a) airway-pressure rescaled-range fluctuation and (b) WBC for AKI; (c) processed PLETH mean for ICU admission. Lines show fold-specific raw log-odds relative to the labeled reference, before recalibration; histograms show finite values in the first training fold. WBC, white blood cell count; PLETH, plethysmography.'
 used=['responses','support','extended_responses']
 details={}
 last_term='PLETH processed mean'
 if replacement is not None:
  caption=('Panels show (a) airway-pressure rescaled-range fluctuation and (b) white blood cell count (WBC) for AKI. '
           'Panel c shows mean capnogram CO₂ for ICU admission, averaged over all samples of the filtered '
           'whole-operation waveform. Lines show fold-specific log-odds relative to the labeled reference before '
           'recalibration. Histograms show finite values in the first training fold.')
  used.append('replacement')
  details['replacement_term']=replacement['provenance']
  last_term='Mean capnogram CO₂'
 save(fig,stem,'Figure S10' if replacement is not None else 'Figure S9','Additional EBM functions',caption,data,used,png_only=replacement is not None,support_data_file=stem+'.training_support.csv',folds=5,missing_bin_panels_included=False,recomputed_term_scores=False,**details)
 CHECKS['extended_responses']={'response_rows':len(data),'training_support_rows':len(hist),'selected_terms':['AWP rescaled-range fluctuation','WBC',last_term],'all_observed_value_functions':True,'missing_bin_panels':0,'all_scores_replayed':True}


def finish(revision=False):
 for path,digest in BEFORE.items():assert sha(ROOT/path)==digest,path
 thumbnails=[]
 for f in MANIFEST:
  p=OUT/(f['stem']+'.png')
  with Image.open(p) as im:
   im=im.convert('RGB');gray=ImageOps.grayscale(im);gray.thumbnail((1028,1400));gray.save(QA/(f['stem']+'.grayscale.png'))
   im.thumbnail((1000,1400));tile=Image.new('RGB',(1040,1460),'white');tile.paste(im,((1040-im.width)//2,45));ImageDraw.Draw(tile).text((15,15),f['label']+' · '+f['title'],fill='black');thumbnails.append(tile)
 sheet=Image.new('RGB',(2080,1460*((len(thumbnails)+1)//2)),ps.color('grid'))
 for i,tile in enumerate(thumbnails):sheet.paste(tile,((i%2)*1040,(i//2)*1460))
 sheet.save(QA/'narrative_contact_sheet.jpg',quality=ps.JPEG_QUALITY)
 report={'status':'passed','scientific_data_unchanged':True,'source_sha256':BEFORE,'figures':CHECKS,'models_fitted':0,'bootstrap_draws_generated':0,'font_range_pt':[8,9],'text_bounds_pass':True}
 if not revision:
  subprocess.run(['pdfunite',*[str(OUT/(f['stem']+'.pdf')) for f in MANIFEST],str(QA/'native_size_figures.pdf')],check=True)
  report['vector_svg_has_no_raster']=True
 (QA/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
 (OUT/'figure_manifest.json').write_text(json.dumps({'scientific_data_policy':'Exact selection and display of saved fitted-model results; no fitting or bootstrap resampling. Native importance means summarize the saved five folds.','figures':MANIFEST},indent=2)+'\n')
 if not revision:
  (OUT/'README.txt').write_text('Narrative-led main-figure revision. Figure 1 remains the previously reviewed cohort flow.\nFigure 2: paired AP/AUROC increments; Figure 3: held-out calibration; Figure 4: native global EBM importance; Figure 5: selected native EBM functions.\nOriginal figures and all source data remain unchanged. This directory supplies SVG/PDF/EPS, PNG600/TIFF600/JPG300, captions, plotted data, and source fingerprints.\nRebuild from the complete workspace with: python 05_CODE/rebuild_narrative_figures.py\n')
def main():
 global ROOT,OUT,QA,BEFORE,SOURCES
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--workspace',type=Path,default=ROOT)
 parser.add_argument('--ebm-revision-output',type=Path)
 parser.add_argument('--include-retired-response',action='store_true',
                     help='Also rebuild the archived ICU capnogram response in revised mode')
 args=parser.parse_args()
 if args.include_retired_response and args.ebm_revision_output is None:
  parser.error('--include-retired-response requires --ebm-revision-output')
 source_paths={key:path.relative_to(ROOT) for key,path in SOURCES.items()}
 ROOT=args.workspace.resolve()
 OUT=(args.ebm_revision_output or ROOT/'03_FIGURES/narrative_revision').resolve()
 QA=OUT/'qa'
 QA.mkdir(parents=True,exist_ok=True)
 SOURCES={key:ROOT/path for key,path in source_paths.items()}
 if args.ebm_revision_output:
  SOURCES.pop('metrics')
  SOURCES.pop('calibration')
  if not args.include_retired_response:
   SOURCES.pop('extended_responses')
  SOURCES['replacement']=OUT/'icu_ebm_response_sources.json'
 BEFORE={path_label(path):sha(path) for path in SOURCES.values()}
 if args.ebm_revision_output:
  replacements=json.loads(SOURCES['replacement'].read_text())['panels']
  importance(revision=True)
  responses(replacements['figure5c'])
  if args.include_retired_response:
   extended_responses(replacements['figure_s10c'])
 else:
  incremental()
  calibration()
  importance()
  responses()
  extended_responses()
 finish(revision=bool(args.ebm_revision_output))


if __name__=='__main__':main()
