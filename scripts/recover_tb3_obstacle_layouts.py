#!/usr/bin/env python3
"""Recover legacy static box layouts from calibrated video, rejecting ambiguity.

Only the recorded brown cuboids at the five configured Isaac candidate sites
are supported. This is image-derived geometry, not simulator ground truth.
Every frame must agree; unknown brown pixels, a moved box, or camera mismatch
rejects the episode. New collections should save simulator geometry directly.
"""
import io
import json
import tarfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROFILE = {
    'camera_height_m': 5.0, 'ground_width_m': 3.048,
    'obstacle_size_m': [.3, .3, 1.0],
    'candidate_sites': [[-.6,.6],[.6,.6],[.6,-.6],[0,.8],[.8,0]],
    'expected_obstacles': 3,
    'provenance': 'turtlebot3_hpc_obs_env_cfg.py camera/obstacle configuration; independently checked against all recorded overhead frames',
}


def hull(points):
    points = sorted(set(points))
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0:
            upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]


def recover_frame(payload):
    image = Image.open(io.BytesIO(payload)).convert('RGB')
    n = image.width
    if image.height != n or n not in (128,144,512):
        raise ValueError('unsupported historical camera canvas')
    rgb = np.asarray(image, dtype=np.int16)
    brown = (rgb[:,:,0]-rgb[:,:,1]>30)&(rgb[:,:,1]-rgb[:,:,2]>35)&(rgb[:,:,0]>100)
    scale = n / PROFILE['ground_width_m']
    center = (n-1)/2
    def project(x,y,z):
        s=scale*5/(5-z)
        return center+s*x, center-s*y
    occupied, scores = [], []
    allowed = Image.new('L',(n,n))
    draw = ImageDraw.Draw(allowed)
    for index,(x,y) in enumerate(PROFILE['candidate_sites']):
        u,v=project(x,y,1)
        radius=max(2,int(.075*scale*1.25))
        ui,vi=round(u),round(v)
        score=float(brown[vi-radius:vi+radius+1,ui-radius:ui+radius+1].mean())
        scores.append(score)
        if score >= .9:
            occupied.append(index)
            corners=[project(x+dx,y+dy,z) for dx in (-.15,.15) for dy in (-.15,.15) for z in (0,1)]
            draw.polygon(hull(corners),fill=255)
        elif score > .03:
            raise ValueError('ambiguous occupied candidate site')
    if len(occupied) != PROFILE['expected_obstacles']:
        raise ValueError('camera/layout mismatch: expected three visible cuboids')
    expanded=np.asarray(allowed.filter(ImageFilter.MaxFilter(5)))>0
    unexpected=float((brown & ~expanded).sum()/max(1,brown.sum()))
    if unexpected > .015:
        raise ValueError('brown geometry outside the calibrated cuboid projections')
    return occupied, {'candidate_occupancy':scores, 'unexpected_brown_fraction':unexpected,
                      'brown_pixels':int(brown.sum()), 'canvas':n}


def recover_shard(path, relative):
    episodes={}
    with tarfile.open(path,'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.overhead.png'):
                prefix=member.name[:-len('.overhead.png')]
                episode=prefix.rsplit('_step',1)[0]
                occupied,audit=recover_frame(tar.extractfile(member).read())
                row=episodes.setdefault(episode, {'occupied_sites':occupied,'frames_checked':0,
                    'maximum_unexpected_brown_fraction':0.0,'minimum_occupied_score':1.0,
                    'maximum_empty_score':0.0,'canvas':audit['canvas']})
                if occupied != row['occupied_sites'] or audit['canvas'] != row['canvas']:
                    raise ValueError('layout changes within a recorded episode')
                row['frames_checked']+=1
                row['maximum_unexpected_brown_fraction']=max(row['maximum_unexpected_brown_fraction'],audit['unexpected_brown_fraction'])
                row['minimum_occupied_score']=min(row['minimum_occupied_score'], min(audit['candidate_occupancy'][i] for i in occupied))
                row['maximum_empty_score']=max(row['maximum_empty_score'], max(audit['candidate_occupancy'][i] for i in range(5) if i not in occupied))
    if not episodes:
        raise ValueError('no image evidence for obstacle reconstruction')
    layouts={}
    for episode,audit in episodes.items():
        if audit['frames_checked']<2:
            raise ValueError('insufficient independent frames for layout reconstruction')
        layouts[f'{relative}::{episode}']={'obstacles':[{'pos':PROFILE['candidate_sites'][i],
            'size_m':PROFILE['obstacle_size_m'][:2], 'yaw':0.0} for i in audit['occupied_sites']],
            'geometry_source':'calibrated_video_candidate_sites_v1','audit':audit}
    return layouts


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shard')
    parser.add_argument('--relative-path',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from pathlib import Path
    Path(args.output).write_text(json.dumps(recover_shard(args.shard,args.relative_path),indent=2)+'\n')
