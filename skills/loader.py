import os
import importlib.util
import importlib
import sys
import yaml
from typing import List, Dict, Type, Optional
from core.engine import CoreEngine
from core.types import SkillManifest  # Update import
from skills.base import BaseSkill
import logging

logger = logging.getLogger("skills.loader")

async def load_skills(engine: CoreEngine, skills_dir: str = "skills"):
    """
    Dynamically load skills from the skills directory using Manifests.
    """
    # Get absolute path to skills directory
    # Assuming this file is in skills/loader.py, so parent is skills, parent of that is root
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_skills_path = os.path.join(base_path, skills_dir)
    
    if not os.path.exists(full_skills_path):
        logger.warning(f"Skills directory not found: {full_skills_path}")
        return

    logger.info(f"Loading skills from: {full_skills_path}")

    for item in os.listdir(full_skills_path):
        item_path = os.path.join(full_skills_path, item)
        if os.path.isdir(item_path) and not item.startswith("__"):
            
            # 1. Check for Manifest
            manifest_path = os.path.join(item_path, "skill.yaml")
            if os.path.exists(manifest_path):
                await _load_skill_with_manifest(engine, item_path, manifest_path)
            else:
                # Legacy Fallback
                logger.warning(f"No skill.yaml found for '{item}'. Attempting legacy load.")
                handler_path = os.path.join(item_path, "handler.py")
                if os.path.exists(handler_path):
                    await _load_skill_from_module(engine, item, handler_path)

async def _load_skill_with_manifest(engine: CoreEngine, skill_dir: str, manifest_path: str):
    """Load a skill based on its manifest."""
    try:
        with open(manifest_path, 'r') as f:
            data = yaml.safe_load(f)
        
        manifest = SkillManifest(**data)
        
        # 2. Dependency Check
        if not _check_requirements(manifest.requires):
            logger.error(f"Skipping skill '{manifest.name}': Missing dependencies {manifest.requires}")
            return

        # 3. Load Module
        entrypoint = manifest.entrypoint
        module_path = os.path.join(skill_dir, entrypoint)
        
        if not os.path.exists(module_path):
            logger.error(f"Entrypoint not found: {module_path}")
            return

        spec = importlib.util.spec_from_file_location(f"skills.{manifest.name}", module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"skills.{manifest.name}"] = module
            spec.loader.exec_module(module)
            
            # Find Skill Class
            skill_class = _find_skill_class(module, manifest.class_name)
            if skill_class:
                try:
                    # 5. DI Injection (Constructor) using engine.services
                    # Simple heuristic for now: default constructor
                    skill_instance = skill_class() 
                    skill_instance.manifest = manifest
                    # Ensure name matches manifest
                    # We might need to monkey-patch name if the class defaults to something else
                    # skill_instance.name = manifest.name 
                    
                    # 3. Sandbox Flag
                    if manifest.requires_sandbox:
                        skill_instance.is_sandboxed = True # Flag for Engine, though Agent doesn't support sandbox execution logic yet
                        logger.info(f"Skill '{manifest.name}' marked for SANDBOX execution.")

                    engine.register_skill(skill_instance)
                    logger.info(f"Loaded skill: {manifest.name} (v{manifest.version})")
                    
                except Exception as e:
                     logger.error(f"Failed to instantiate {manifest.name}: {e}")
            else:
                logger.error(f"No BaseSkill subclass found in {manifest.name}")

    except Exception as e:
        logger.error(f"Failed to load manifest for {skill_dir}: {e}")

def _check_requirements(reqs: List[str]) -> bool:
    """Check if required python packages are installed."""
    missing = []
    for pkg in reqs:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.warning(f"Missing requirements: {missing}")
        return False
    return True

def _find_skill_class(module, class_name: str = None) -> Type[BaseSkill]:
    """Find the BaseSkill subclass in the module."""
    import inspect
    
    if class_name:
        if hasattr(module, class_name):
            return getattr(module, class_name)
        return None

    # Auto-detect
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BaseSkill) and obj is not BaseSkill:
            return obj
    return None

async def _load_skill_from_module(engine: CoreEngine, item_name: str, handler_path: str):
    """
    Imports a module and finds the BaseSkill subclass to instantiate.
    Legacy method for backward compatibility.
    """
    spec = importlib.util.spec_from_file_location(f"skills.{item_name}", handler_path)
    if spec and spec.loader:
        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"skills.{item_name}"] = module
            spec.loader.exec_module(module)
            
            # Find BaseSkill subclass
            import inspect
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseSkill) and obj is not BaseSkill:
                    # Legacy DI check
                    sig = inspect.signature(obj.__init__)
                    kwargs = {}
                    
                    try:
                        skill_instance = obj(**kwargs)
                        engine.register_skill(skill_instance)
                        logger.info(f"Loaded legacy skill: {skill_instance.name}")
                    except Exception as e:
                         logger.error(f"Failed to instantiate legacy {obj.__name__}: {e}")
                    break
        except Exception as e:
            logger.error(f"Failed to load legacy skill {item_name}: {e}")
