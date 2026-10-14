"""Run the established shell inference workflow on three exact checkpoints."""
import json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'evaluation_manifest.json').read_text())
paths=[row['episode_tar'] for row in manifest['episodes']]
assert len(paths)==len(set(paths))==100
for family,checkpoint in manifest['checkpoints'].items():
    assert Path(checkpoint).is_file() and checkpoint.endswith('_epoch_'+family[5:]+'.pt')
    env=os.environ.copy()
    env.update(MA_VLCM_SCRATCH_ROOT='/scratch/aparame/ma_vlcm',CONTAINER_PATH='/home/aparame/Research/MA-VLCM/ma_vlcm.sif',
        HF_HOME='/scratch/aparame/VLCM/ma_vlcm/cache/huggingface',HF_HUB_CACHE='/scratch/aparame/VLCM/ma_vlcm/cache/huggingface/hub',
        TRANSFORMERS_CACHE='/scratch/aparame/VLCM/ma_vlcm/cache/huggingface/hub',MPLCONFIGDIR=str(root/'mpl_cache'),
        OUTPUT_DIR=str(root/'results'),RUN_NAME=family,NUM_EPISODES='100',NUM_FAILED_EPISODES='0',
        QWEN3_VL_CHECKPOINT=checkpoint,NO_VIDEO='1',CLIP_STRIDE='1',OMP_NUM_THREADS='8')
    print('Evaluating',family,'on the same 100 real held-out episodes',flush=True)
    subprocess.run(['bash',str(root/'source/scripts/run_tb3_episode_inference_all.sh'),
        '--models','qwen3_vl','--minimum-frames','8','--episode-shards',*paths],env=env,cwd=root/'source',check=True)
    (root/(family+'_COMPLETE.json')).write_text(json.dumps({'checkpoint':checkpoint,'episodes':100,'status':'completed'},indent=2))
subprocess.run(['apptainer','exec','-B','/scratch:/scratch','-B','/home:/home',
    '/home/aparame/Research/MA-VLCM/ma_vlcm.sif','python3',str(root/'plot_comparison.py'),str(root)],check=True)
(root/'EVALUATION_COMPLETE.json').write_text(json.dumps({'status':'completed','epochs':[22,63,75],'episodes':100,
    'metrics':str(root/'metrics_summary.json'),'plot':str(root/'comparison_overview.png')},indent=2)+'\n')
