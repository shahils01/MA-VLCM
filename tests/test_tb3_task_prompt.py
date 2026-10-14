"""Task text cannot change when outcome/state labels change."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import warnings
import pytest
from ma_vlcm.tb3_task_prompt import (
    TEXT_INPUT_CONTRACT, build_tb3_task_prompt, tb3_task_context,
    validate_task_instruction, warn_legacy_text_contract,
)


def scene(domain='static_obstacles'):
    return {'task_domain':domain,'num_agents':3,'num_static_obstacles':3,
            'agents':[{'id':i,'pos':[.2*i,0.],'goal_pos':[1.,1.],
                       'dist_to_goal':1.,'initial_goal_distance':2.,
                       'collision':False,'reached':False} for i in range(3)]}


def test_task_instruction_is_invariant_to_all_frame_state_and_supervision():
    state=scene();expected=build_tb3_task_prompt(state)
    changed=copy.deepcopy(state)
    changed.update(progress={'target':-1},reward=-10,goal_cost=.9,
                   agent_risk=1.,obstacle_risk=1.,synthetic_rewind=True,
                   episode_meta={'step':999,'success':True,'failure':True,
                                 'outcome':'success','termination_reason':'collision'})
    for agent in changed['agents']:
        agent.update(pos=[999,-999],goal_pos=[555,555],dist_to_goal=0,
                     initial_goal_distance=30,collision=True,reached=True)
    assert build_tb3_task_prompt(changed)==expected
    assert 'assigned goal' in expected and 'static obstacles' in expected
    assert not any(x in expected for x in ['999','555','reached:','collision:','potential'])


def test_domain_and_cardinality_still_describe_the_task():
    ao=scene();gtg=scene('goal_to_goal')
    assert build_tb3_task_prompt(ao)!=build_tb3_task_prompt(gtg)
    assert tb3_task_context(ao)[:3]==('static_obstacles',3,3)
    assert tb3_task_context(gtg)[:3]==('goal_to_goal',3,0)
    ao['num_agents']=4
    assert 'exactly 4 TurtleBot3 agents' in build_tb3_task_prompt(ao)


def test_no_frame_metadata_can_enter_via_stored_instruction_or_task_file():
    state=scene();expected=build_tb3_task_prompt(state)
    state['task_instruction']='Avoid static obstacles. collision: yes; reward: 99'
    assert build_tb3_task_prompt(state,{'task.txt':b'collision: yes'})==expected


@pytest.mark.parametrize('text',[
    'collision: yes','reached: no','success=True','failure: false',
    'done: 1','reward: -1','target: .5','raw_target: .1','obstacle_risk: .9',
    'agent_risk: .2','progress: .9','Timestep: 5','Episode outcome token: success',
    'termination_reason: contact','dist_to_goal: .2','position: (1,2)',
    'Navigate safely. {collision}','',
])
def test_state_augmented_custom_overrides_are_rejected(text):
    with pytest.raises(ValueError):build_tb3_task_prompt(scene(),task_instruction=text)


def test_literal_custom_task_is_not_augmented():
    task='Move the red robot to goal A while avoiding obstacles.'
    assert build_tb3_task_prompt(scene(),task_instruction=task)==task
    assert validate_task_instruction(task)==task


def extracted_function(path,name):
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace={'build_tb3_task_prompt':build_tb3_task_prompt}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def test_actual_offline_wrapper_cannot_serialize_legacy_metadata():
    root=Path(__file__).resolve().parents[1]
    builder=extracted_function(root/'src/ma_vlcm/train.py','_build_tb3_prompt')
    state=scene()
    assert builder(state,{},['collision: yes; reached: yes'],999,'success','collision',target_schema='tb3_potential_v1')==build_tb3_task_prompt(state)


def test_live_adapter_uses_identical_task_text():
    root=Path(__file__).resolve().parents[1]
    builder=extracted_function(root/'src/ma_vlcm/tb3_live_inference.py','_build_prompt')
    assert builder(SimpleNamespace(),scene())==build_tb3_task_prompt(scene())


def test_old_checkpoint_warning_cannot_silently_claim_clean_evaluation():
    with pytest.warns(UserWarning,match='distribution-shift diagnostic'):
        assert warn_legacy_text_contract({})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert not warn_legacy_text_contract(SimpleNamespace(text_input_contract=TEXT_INPUT_CONTRACT))
    assert not caught
