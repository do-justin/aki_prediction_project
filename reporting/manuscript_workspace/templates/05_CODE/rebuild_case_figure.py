"""Rebuild the held-out EBM illustration from portable contribution CSVs only."""

import publication_style as ps
from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'03_FIGURES/heldout_cases'
STEM='heldout_ebm_cases'
CAPTION='Held-out EBM explanations for cases nearest the 10th and 90th percentiles of combined-model AKI or ICU-admission risk. Selection did not use observed outcomes. Filled and open bars raise and lower calibrated log-odds; gray and green bars show the calibrated model intercept and final prediction. Clinical contributions are grouped by feature; waveform contributions by channel. Headers give clinical-only and combined risks. WBC, white blood cell count (10³/µL); BUN, blood urea nitrogen (mg/dL); Hb, hemoglobin (g/dL); ASA, American Society of Anesthesiologists physical status.'
REVISION_CAPTION=(
 'Cases are nearest the 10th and 90th percentiles of combined-model AKI or ICU admission risk. Selection did not '
 'use observed outcomes. Filled and open bars raise and lower calibrated log-odds. Gray and green bars show '
 'the calibrated model intercept and final prediction. Preoperative contributions are grouped by feature and '
 'waveform contributions by channel. Headers give preoperative-only and combined risks. WBC, white blood cell '
 'count (10³/µL). BUN, blood urea nitrogen (mg/dL). Hb, hemoglobin (g/dL). ASA-PS, American Society of '
 'Anesthesiologists physical status.'
)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def build(source, output, revision=False):
 source_hashes={name:sha(source/name) for name in (
  'case_selection_ledger.csv','case_clinical_context.csv','grouped_calibrated_contributions.csv'
 )}
 cases=pd.read_csv(source/'case_selection_ledger.csv')
 groups=pd.read_csv(source/'grouped_calibrated_contributions.csv')
 context=pd.read_csv(source/'case_clinical_context.csv').set_index(['display_case','feature'])
 stem='S11_EBM_heldout_cases' if revision else STEM
 output.mkdir(parents=True,exist_ok=True)
 assert len(cases)==4 and len(groups)==32
 assert not cases.selection_consulted_outcome.any()
 for row in cases.itertuples():
  data=groups[groups.display_case.eq(row.display_case)].sort_values('order')
  assert abs(row.calibrated_baseline+data.contribution.sum()-row.calibrated_final_logit)<1e-12
  assert abs(1/(1+np.exp(-row.calibrated_final_logit))-row.combined_probability)<1e-12
 values=np.r_[groups.start.to_numpy(),groups.end.to_numpy(),cases.calibrated_baseline.to_numpy(),cases.calibrated_final_logit.to_numpy(),0]
 low,high=values.min(),values.max();span=high-low;xlim=(low-.10*span,high+.10*span)
 with ps.style_context(**{'axes.spines.top': False, 'axes.spines.right': False}):
  fig=plt.figure(figsize=(ps.WIDTH_MM/25.4,215/25.4));grid=fig.add_gridspec(2,2,left=.155,right=.985,top=.890,bottom=.155,wspace=.62,hspace=.56)
  for i,row in enumerate(cases.itertuples()):
   ax=fig.add_subplot(grid[i//2,i%2])
   data=groups[groups.display_case.eq(row.display_case)].sort_values('order')
   short={'White blood cell count':'WBC','Blood urea nitrogen':'BUN','ASA physical status':'ASA class','Surgical department':'Department','Other clinical terms':'Other clinical'}
   if revision:
    short.update({'Other clinical terms':'Other\npreoperative','PLETH':'PPG',
                  'CO₂':'Capnography','AWP':'Airway pressure','ASA physical status':'ASA-PS'})
   clinical_labels=[]
   for item in data.itertuples():
    if item.order>3:clinical_labels.append(short.get(item.label,item.label));continue
    value=context.loc[(row.display_case,item.group),'raw_value']
    if item.group=='preop_wbc':label='WBC: '+('missing' if pd.isna(value) else f'{float(value):.2f}')
    elif item.group=='preop_hb':label=f'Hb: {float(value):.1f}'
    elif item.group=='preop_bun':label=f'BUN: {float(value):.0f}'
    elif item.group=='age':label=f'Age: {float(value):.0f} years'
    elif item.group=='asa':
     label=('ASA-PS: ' if revision else 'ASA: ')+{1:'I',2:'II',3:'III',4:'IV'}.get(int(float(value)),str(value))
    elif item.group=='optype':label='Procedure:\n'+str(value)
    elif item.group=='department':label='Department:\n'+str(value)
    else:label=short.get(item.label,item.label)
    clinical_labels.append(label)
   labels=['Intercept']+clinical_labels+['Prediction'];baseline=row.calibrated_baseline;final=row.calibrated_final_logit
   positions=[0,1.4,2.8,4.2,5.4,6.4,7.4,8.4,9.4,10.5]
   ax.barh(0,baseline,left=0,height=.64,facecolor=ps.color('baseline'),edgecolor=ps.color('baseline_edge'),linewidth=.65,zorder=3)
   for j,item in enumerate(data.itertuples(),1):
    start,end=item.start,item.end
    ax.plot([start,start],[positions[j-1]+.34,positions[j]-.34],color=ps.color('connector'),lw=.65,zorder=2)
    ax.barh(positions[j],abs(end-start),left=min(start,end),height=.64,facecolor=ps.color('signed_contribution') if end>=start else 'white',edgecolor=ps.color('signed_contribution'),linewidth=.85,zorder=3)
   ax.plot([final,final],[positions[-2]+.34,positions[-1]-.34],color=ps.color('connector'),lw=.65,zorder=2)
   ax.barh(positions[-1],final,left=0,height=.64,facecolor=ps.INPUT_COLORS['preop_and_all_waveforms'],edgecolor=ps.color('combined_edge'),linewidth=.65,zorder=3)
   ax.set(yticks=positions,yticklabels=labels,ylim=(positions[-1]+.55,-.65),xlim=xlim)
   ticks=MaxNLocator(4).tick_values(*xlim);ax.set_xticks([tick for tick in ticks if xlim[0]<=tick<=xlim[1]]);ax.tick_params(axis='y',length=0,pad=3)
   ax.axvline(0,color=ps.color('zero'),lw=ps.REFERENCE_WIDTH,zorder=1);ax.grid(axis='x',color=ps.color('grid'),linewidth=ps.GRID_WIDTH,zorder=0)
   ax.set_xlabel('Calibrated log-odds',labelpad=4)
   title=r'$\bf{'+chr(97+i)+'}$  '+f'{row.outcome_label} · {int(row.target_quantile*100)}th percentile'
   ax.text(0,1.20,title,transform=ax.transAxes,fontsize=ps.STYLE['figure']['title_pt'],fontweight='normal',va='bottom')
   preoperative_percent=f'{row.clinical_probability:.1%}'
   combined_percent=f'{row.combined_probability:.1%}'
   status='Yes' if row.observed_outcome else 'No'
   if revision:
    summary=f'Preoperative only {preoperative_percent}\nCombined {combined_percent}  |  Outcome: {status}'
   else:
    summary=f'Clinical {preoperative_percent}  |  Combined {combined_percent}\nOutcome: {status}'
   ax.text(0,1.07,summary,transform=ax.transAxes,fontsize=ps.STYLE['figure']['body_pt'],va='bottom',linespacing=1.25)
   # Terminal probability is placed outside the plotting box in the panel key,
   # rather than mislabeled as an additive probability contribution.
  fig.legend(handles=[Patch(facecolor=ps.color('signed_contribution'),edgecolor=ps.color('signed_contribution'),label='Raises risk'),Patch(facecolor='white',edgecolor=ps.color('signed_contribution'),label='Lowers risk'),Patch(facecolor=ps.color('baseline'),edgecolor=ps.color('baseline_edge'),label='Intercept'),Patch(facecolor=ps.INPUT_COLORS['preop_and_all_waveforms'],edgecolor=ps.color('combined_edge'),label='Prediction')],loc='lower center',bbox_to_anchor=(.5,.010),ncol=2,frameon=False,handlelength=1.5,columnspacing=2.0)
  fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
  for text in fig.findobj(matplotlib.text.Text):
   if not text.get_visible() or not text.get_text().strip():continue
   bbox=text.get_window_extent(renderer)
   if bbox.x0<-.5 or bbox.y0<-.5 or bbox.x1>fig.bbox.width+.5 or bbox.y1>fig.bbox.height+.5:outside.append(text.get_text())
  assert not outside,('off-canvas text',outside)
  formats = ['png'] if revision else ['svg', 'pdf', 'eps', 'png', 'tiff', 'jpg']
  if not revision:
   for extension in ('svg', 'pdf', 'eps'):
    fig.savefig(output/(stem+'.'+extension), format=extension)
  fig.savefig(output/(stem+'.png'),dpi=ps.PNG_DPI)
  if not revision:
   with Image.open(output/(stem+'.png')) as raster:
    png = raster.convert('RGB')
   fig.savefig(output/(stem+'.tiff'), dpi=ps.TIFF_DPI, pil_kwargs={'compression':ps.TIFF_COMPRESSION})
   with Image.open(output/(stem+'.tiff')) as raster:
    tiff = raster.convert('RGB')
   tiff.save(output/(stem+'.tiff'), compression=ps.TIFF_COMPRESSION, dpi=(ps.TIFF_DPI,ps.TIFF_DPI))
   jpg = png.resize((round(ps.WIDTH_MM/25.4*ps.JPG_DPI), round(215/25.4*ps.JPG_DPI)), Image.Resampling.LANCZOS)
   jpg.save(output/(stem+'.jpg'), quality=ps.JPEG_QUALITY, dpi=(ps.JPG_DPI,ps.JPG_DPI))
  plt.close(fig)
 (output/(stem+'.caption.txt')).write_text((REVISION_CAPTION if revision else CAPTION)+'\n')
 groups.to_csv(output/(stem+'.plotted.csv'),index=False)
 assert all(sha(source/name)==digest for name,digest in source_hashes.items())
 checks={'status':'passed','case_count':4,'contribution_groups':len(groups),'prediction_reconstruction_tolerance':1e-12,'off_canvas_labels':0,'shared_axis_limits_calibrated_logit':list(xlim),'displayed_case_identifiers':'Case A–D only; original IDs retained in selection ledger','data_sha256':source_hashes,'models_fitted':0}
 if not revision:
  checks['svg_has_no_raster_images'] = '<image' not in (output/(stem+'.svg')).read_text()
 (output/('S11_figure_validation.json' if revision else 'figure_validation.json')).write_text(json.dumps(checks,indent=2)+'\n')
 item={'style_profile_sha256':ps.STYLE_SHA256,'stem':stem,'title':'EBM case explanations','label':'Figure S10','width_mm':ps.WIDTH_MM,'height_mm':215,'caption_file':stem+'.caption.txt','plotted_data_file':stem+'.plotted.csv','figure_role':'supplementary illustration' if revision else 'supplementary illustration candidate','formats':formats,'font_range_pt':[ps.STYLE['figure']['body_pt'],ps.STYLE['figure']['title_pt']]}
 (output/('S11_figure_manifest.json' if revision else 'figure_manifest.json')).write_text(json.dumps({'figures':[item],'portable_rebuild':'05_CODE/rebuild_case_figure.py','checks':checks},indent=2)+'\n')
 print('Held-out case figure built and exact reconstruction checks passed')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--source',type=Path,default=OUT)
 parser.add_argument('--revision-output',type=Path)
 args=parser.parse_args()
 build(args.source,args.revision_output or OUT,revision=bool(args.revision_output))
