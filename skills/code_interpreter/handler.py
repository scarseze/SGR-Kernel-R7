import asyncio
import docker
import tarfile
import io
import os
from typing import Type
from pydantic import BaseModel

from core.state import AgentState
from skills.base import BaseSkill, SkillMetadata
from skills.code_interpreter.schema import CodeExecutionRequest, CodeExecutionResult

class CodeInterpreterSkill(BaseSkill):
    name: str = "code_interpreter"
    description: str = (
        "Executes Python code in a secure sandboxed environment. "
        "Use this for data analysis, math, or testing code snippets."
    )
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            capabilities=["python_execution", "sandbox", "math", "data_analysis"],
            risk_level="high",
            side_effects=False,
            idempotent=True,
            requires_network=False,
            requires_filesystem=True,
            cost_class="expensive"
        )
    
    # Docker configuration
    IMAGE_NAME = "sgr-sandbox:latest"
    CONTAINER_NAME = "sgr-sandbox-instance"
    
    def __init__(self, **data):
        super().__init__(**data)
        try:
            self.client = docker.from_env()
            self.client.ping()  # Verify connection
        except (docker.errors.DockerException, Exception) as e:
            # Re-raise with a user-friendly message
            raise RuntimeError(
                f"[{self.name}] Failed to connect to Docker. "
                "Please ensure Docker Desktop is running. "
                f"Error: {e}"
            ) from e
        self._ensure_image_exists()
        self._start_sandbox()

    @property
    def input_schema(self) -> Type[BaseModel]:
        return CodeExecutionRequest

    def _ensure_image_exists(self):
        try:
            self.client.images.get(self.IMAGE_NAME)
        except docker.errors.ImageNotFound:
            print(f"[{self.name}] Building sandbox image...")
            # Assuming Dockerfile is in the same directory as handler.py
            dockerfile_path = os.path.dirname(os.path.abspath(__file__))
            self.client.images.build(path=dockerfile_path, tag=self.IMAGE_NAME, filename="Dockerfile.sandbox")
            print(f"[{self.name}] Sandbox image built.")

    def _start_sandbox(self):
        # Check if container is already running
        try:
            container = self.client.containers.get(self.CONTAINER_NAME)
            if container.status != "running":
                container.restart()
        except docker.errors.NotFound:
            self.client.containers.run(
                self.IMAGE_NAME,
                name=self.CONTAINER_NAME,
                detach=True,
                mem_limit="512m",  # Limit memory
                nano_cpus=500000000,  # Limit CPU (0.5 CPU)
                # network_disabled=True # ENABLE NETWORK for RLM Proxy
                extra_hosts={"host.docker.internal": "host-gateway"} # Allow access to host
            )

    async def execute(self, params: CodeExecutionRequest, state: AgentState) -> CodeExecutionResult:
        container = self.client.containers.get(self.CONTAINER_NAME)
        
        # Prepare code script
        script_content = params.code.encode('utf-8')
        
        # We need to copy the script into the container
        # Docker API put_archive needs a tar stream
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            tar_info = tarfile.TarInfo(name='script.py')
            tar_info.size = len(script_content)
            tar.addfile(tar_info, io.BytesIO(script_content))
        tar_stream.seek(0)
        
        container.put_archive('/home/sanduser', tar_stream)
        
        # Execute the script
        # We wrap it in python execution
        exec_cmd = f"python /home/sanduser/script.py"
        
        try:
            # Running with timeout mechanism needs careful handling with docker exec
            # docker-py exec_run is synchronous usually, so we run it in thread pool or use async adapter
            # For MVP we use synchronous call but verify timeout logic in future
            
            # Run blocking docker exec in a separate thread to avoid freezing the async loop
            loop = asyncio.get_running_loop()
            exec_result = await loop.run_in_executor(
                None, 
                lambda: container.exec_run(
                    exec_cmd,
                    workdir="/home/sanduser",
                    user="sanduser"
                )
            )
            
            return CodeExecutionResult(
                success=(exec_result.exit_code == 0),
                stdout=exec_result.output.decode('utf-8') if exec_result.output else "",
                stderr="", # Docker exec_run combines stdout/stderr by default unless demux used
                exit_code=exec_result.exit_code
            )
            
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1
            )
