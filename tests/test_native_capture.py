from pathlib import Path
import importlib.util
from types import SimpleNamespace
import pytest
spec=importlib.util.spec_from_file_location('native_capture',Path(__file__).parents[1]/'tools/collection/native_capture.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_native_camera_and_output_match_without_changing_actor_shape():
    cfg=SimpleNamespace(global_camera_cfg=SimpleNamespace(width=512,height=512),
        vlcm_camera_image_size=(128,128),camera_image_size=(128,128),actor_camera_image_size=(64,64))
    receipt=module.configure_native_overhead(cfg,224)
    assert cfg.global_camera_cfg.width==cfg.global_camera_cfg.height==224
    assert cfg.vlcm_camera_image_size==cfg.camera_image_size==(224,224)
    assert cfg.actor_camera_image_size==(64,64) and receipt['upscaled'] is False

def test_unknown_camera_refuses_to_collect():
    with pytest.raises(ValueError,match='overhead camera'):
        module.configure_native_overhead(SimpleNamespace(actor_camera_image_size=(64,64)),224)
