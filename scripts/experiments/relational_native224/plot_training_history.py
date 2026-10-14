import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);rows=json.loads((root/'training_history.json').read_text());epochs=[r['epoch'] for r in rows]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13,8.5))
for ax in axes.flat:
    for epoch,color in [(22,'#16885b'),(63,'#8a4ca8'),(75,'#df7d22')]:ax.axvline(epoch,color=color,alpha=.32,lw=1)
    ax.grid(alpha=.2);ax.set_xlabel('Epoch');ax.set_xlim(1,76)
ax=axes[0,0]
for prefix,color in [('train','#16885b'),('val','#1867b8')]:ax.plot(epochs,[r[prefix+'/epoch_loss'] for r in rows],color=color,label=prefix.title())
ax.set(title='Composite loss: growing train–validation gap',ylabel='Logged composite loss');ax.legend()
ax=axes[0,1]
for prefix,color in [('train','#16885b'),('val','#1867b8')]:ax.semilogy(epochs,[r[prefix+'/epoch_progress_mse_loss'] for r in rows],color=color,label=prefix.title())
ax.set(title='Scalar potential error: best validation at epoch 63',ylabel='Logged MSE (log scale)');ax.legend()
ax=axes[1,0]
for prefix,color in [('train','#16885b'),('val','#1867b8')]:ax.plot(epochs,[r[prefix+'/epoch_categorical_ce_loss'] for r in rows],color=color,label=prefix.title())
ax.set(title='Categorical CE: best validation at epoch 22',ylabel='Logged categorical CE');ax.legend()
ax=axes[1,1]
for key,color,label in [('val/drop_direction_recall','#ba4b3b','Drop direction recall'),('val/recovery_direction_recall','#16885b','Recovery direction recall')]:ax.plot(epochs,[r[key] for r in rows],color=color,label=label)
ax.set(title='Reward-change direction on validation clips',ylabel='Recall',ylim=(.35,.85));ax.legend()
fig.suptitle('Relational native224 Qwen · completed 75 epochs',fontsize=16)
fig.text(.5,.016,'Green marker: epoch 22 (best total / CE) · Purple: epoch 63 (best scalar MSE) · Orange: epoch 75 (final)\nTraining validation logs; these are not the new full-episode inference results.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.055,1,.95]);fig.savefig(root/'training_history.png',dpi=165);plt.close(fig)
summary={'status':'training_finished','epochs_completed':75,'best_composite_epoch':22,'best_scalar_mse_epoch':63,'comparison_epochs':[22,63,75],
    'selected_epoch_metrics':[{k:v for k,v in row.items() if k=='epoch' or k in ['train/epoch_loss','val/epoch_loss','val/epoch_progress_mse_loss','val/epoch_categorical_ce_loss','val/change_mae','val/drop_direction_recall','val/recovery_direction_recall','val/trajectory_preference_accuracy']} for row in rows if row['epoch'] in [22,63,75]],
    'interpretation':'Composite/CE generalization gap after epoch22; scalar and temporal errors keep improving later. Epoch75 scalar MSE is 13.71% worse than epoch63. Offline matched episode inference is required before choosing the critic.'}
(root/'training_analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
